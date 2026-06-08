"""
End-to-end inference: localize (xywh) -> mask -> classify (multi-backbone ensemble).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms

from . import apply_mask_to_image, load_ensemble_from_dir
from .localizer import Detection, LocateAnythingWrapper

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def preprocess(image: Image.Image, size: int = 224) -> torch.Tensor:
    t = transforms.Compose([
        transforms.Resize(int(size * 1.14)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return t(image).unsqueeze(0)


def crop_by_bbox(image: Image.Image, det: Detection, pad: float = 0.05) -> Image.Image:
    w, h = image.size
    x1 = int(max(0, (det.x - pad) * w))
    y1 = int(max(0, (det.y - pad) * h))
    x2 = int(min(w, (det.x + det.w + pad) * w))
    y2 = int(min(h, (det.y + det.h + pad) * h))
    return image.crop((x1, y1, x2, y2))


class AnimalRecognitionPipeline:
    def __init__(
        self,
        weights_dir: str | Path,
        device: str | None = None,
        use_locateanything: bool = True,
        use_trained_localizer: bool = True,
        classifier_backbones: tuple[str, ...] | None = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt_dir = Path(weights_dir)
        self.classifier_backbones = classifier_backbones

        with open(ckpt_dir / "class_names.json", encoding="utf-8") as f:
            self.class_names: List[str] = json.load(f)
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

        from .ensemble import CLASSIFIER_BACKBONES

        backbones = classifier_backbones or CLASSIFIER_BACKBONES
        self.ensemble = load_ensemble_from_dir(
            ckpt_dir,
            len(self.class_names),
            device=self.device,
            backbones=backbones,
        )
        if self.ensemble is None:
            from . import build_classifier

            cls_ckpt = torch.load(
                ckpt_dir / "classifier_cbam_best.pth",
                map_location=self.device,
                weights_only=False,
            )
            self.classifier = build_classifier(len(self.class_names), pretrained=False)
            self.classifier.load_state_dict(cls_ckpt["state_dict"])
            self.classifier.to(self.device).eval()
        else:
            self.classifier = None

        self.use_trained_localizer = use_trained_localizer
        self.localizer = None
        loc_path = ckpt_dir / "localizer_best.pth"
        if use_trained_localizer and loc_path.exists():
            from . import load_localizer

            self.localizer, _ = load_localizer(loc_path, device=str(self.device))
            self.localizer.eval()

        self.locateanything = LocateAnythingWrapper() if use_locateanything else None
        if use_locateanything:
            if self.locateanything and self.locateanything.worker is not None:
                print("[Pipeline] Localization: LocateAnything (multi-box)")
            else:
                print("[Pipeline] LocateAnything unavailable; install from NVlabs/Eagle Embodied/")
        elif self.localizer is not None:
            print("[Pipeline] Localization: trained BboxLocalizer (single-box)")

    def localize(self, image: Image.Image) -> List[Detection]:
        categories = self.class_names

        if self.locateanything and self.locateanything.worker is not None:
            dets = self.locateanything.detect_animals(
                image, categories, self.class_to_idx, phrase="animal"
            )
            if dets:
                return dets
            if not self.use_trained_localizer:
                return [Detection(x=0.05, y=0.05, w=0.9, h=0.9, confidence=0.5)]

        if self.localizer is not None:
            tensor = preprocess(image).to(self.device)
            with torch.no_grad():
                bbox = self.localizer(tensor)[0]
            return [Detection(
                x=bbox[0].item(), y=bbox[1].item(),
                w=bbox[2].item(), h=bbox[3].item(),
                confidence=1.0,
            )]

        return [Detection(x=0.05, y=0.05, w=0.9, h=0.9, confidence=0.5)]

    def classify_crop(self, crop: Image.Image) -> tuple[str, float, dict]:
        tensor = preprocess(crop).to(self.device)
        bbox = torch.tensor([[0.0, 0.0, 1.0, 1.0]], device=self.device)
        masked = apply_mask_to_image(tensor, bbox)

        with torch.no_grad():
            if self.ensemble is not None:
                details = self.ensemble.forward_with_details(masked)
                probs = F.softmax(details["logits"], dim=1)[0]
            else:
                logits = self.classifier(masked)
                probs = F.softmax(logits, dim=1)[0]

        idx = probs.argmax().item()
        prob_dict = {self.class_names[i]: probs[i].item() for i in range(len(self.class_names))}
        return self.class_names[idx], probs[idx].item(), prob_dict

    def predict_image(self, image: Image.Image) -> list[dict]:
        detections = self.localize(image)
        results = []

        for det in detections:
            crop = crop_by_bbox(image, det)
            label, conf, all_probs = self.classify_crop(crop)
            results.append({
                "bbox_xywh": [det.x, det.y, det.w, det.h],
                "class": label,
                "confidence": conf,
                "probabilities": all_probs,
                "top5": sorted(all_probs.items(), key=lambda x: -x[1])[:5],
            })

        return results

    def predict(self, image_path: str) -> list[dict]:
        image = Image.open(image_path).convert("RGB")
        return self.predict_image(image)

    def visualize(self, image: Image.Image, results: list[dict]) -> Image.Image:
        out = image.copy()
        draw = ImageDraw.Draw(out)
        w, h = out.size

        for r in results:
            x, y, bw, bh = r["bbox_xywh"]
            x1, y1 = x * w, y * h
            x2, y2 = (x + bw) * w, (y + bh) * h
            draw.rectangle([x1, y1, x2, y2], outline="#00e676", width=3)
            draw.text((x1, max(y1 - 14, 0)), f"{r['class']} {r['confidence']:.2f}", fill="#00e676")

        return out

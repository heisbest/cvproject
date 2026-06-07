"""Animal localizer: trainable bbox head + optional LocateAnything inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18


@dataclass
class Detection:
    """Single detection in xywh format (normalized 0-1)."""

    x: float
    y: float
    w: float
    h: float
    confidence: float
    class_name: str = "animal"
    class_id: int = -1


def xywh_to_xyxy(box: torch.Tensor) -> torch.Tensor:
    x, y, w, h = box.unbind(-1)
    return torch.stack([x, y, x + w, y + h], dim=-1)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """IoU for xywh boxes, shape (N, 4) vs (M, 4)."""
    b1 = xywh_to_xyxy(boxes1)
    b2 = xywh_to_xyxy(boxes2)

    inter_x1 = torch.max(b1[:, None, 0], b2[None, :, 0])
    inter_y1 = torch.max(b1[:, None, 1], b2[None, :, 1])
    inter_x2 = torch.min(b1[:, None, 2], b2[None, :, 2])
    inter_y2 = torch.min(b1[:, None, 3], b2[None, :, 3])

    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-6)


class BboxLocalizer(nn.Module):
    """
    Lightweight localizer predicting primary animal bbox (xywh, sigmoid-normalized).
    Trained separately from the classifier on annotated datasets.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(self.features(x)))


def localization_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Smooth L1 + GIoU on xywh boxes."""
    l1 = F.smooth_l1_loss(pred, target, reduction="mean")

    pred_xyxy = xywh_to_xyxy(pred)
    tgt_xyxy = xywh_to_xyxy(target)

    inter_x1 = torch.max(pred_xyxy[:, 0], tgt_xyxy[:, 0])
    inter_y1 = torch.max(pred_xyxy[:, 1], tgt_xyxy[:, 1])
    inter_x2 = torch.min(pred_xyxy[:, 2], tgt_xyxy[:, 2])
    inter_y2 = torch.min(pred_xyxy[:, 3], tgt_xyxy[:, 3])
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    area_p = (pred_xyxy[:, 2] - pred_xyxy[:, 0]) * (pred_xyxy[:, 3] - pred_xyxy[:, 1])
    area_t = (tgt_xyxy[:, 2] - tgt_xyxy[:, 0]) * (tgt_xyxy[:, 3] - tgt_xyxy[:, 1])
    union = area_p + area_t - inter
    iou = inter / union.clamp(min=1e-6)

    enc_x1 = torch.min(pred_xyxy[:, 0], tgt_xyxy[:, 0])
    enc_y1 = torch.min(pred_xyxy[:, 1], tgt_xyxy[:, 1])
    enc_x2 = torch.max(pred_xyxy[:, 2], tgt_xyxy[:, 2])
    enc_y2 = torch.max(pred_xyxy[:, 3], tgt_xyxy[:, 3])
    enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1)
    giou = iou - (enc_area - union) / enc_area.clamp(min=1e-6)

    return l1 + (1 - giou).mean()


class LocateAnythingWrapper:
    """
    Wrapper for nvidia/LocateAnything-3B (preferred at inference).
    Falls back to trained BboxLocalizer when the model is unavailable.
    Install: pip install locateanything-worker  (from NVlabs/Eagle repo)
    """

    def __init__(self, model_id: str = "nvidia/LocateAnything-3B", device: str = "cuda"):
        self.device = device
        self.worker = None
        self._load_worker(model_id)

    def _load_worker(self, model_id: str) -> None:
        try:
            from locateanything_worker import LocateAnythingWorker

            self.worker = LocateAnythingWorker(model_id)
            print(f"LocateAnything loaded: {model_id}")
        except ImportError:
            print(
                "LocateAnything not installed. Use BboxLocalizer or install from "
                "https://github.com/NVlabs/Eagle (Embodied/locateanything_worker.py)"
            )
        except Exception as exc:
            print(f"LocateAnything load failed: {exc}")

    @staticmethod
    def _xyxy_to_xywh(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> Detection:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        return Detection(
            x=x1 / w,
            y=y1 / h,
            w=(x2 - x1) / w,
            h=(y2 - y1) / h,
            confidence=1.0,
        )

    def detect(
        self,
        image: Image.Image,
        categories: List[str],
        class_to_idx: Optional[dict] = None,
    ) -> List[Detection]:
        if self.worker is None:
            return []

        w, h = image.size
        try:
            from locateanything_worker import LocateAnythingWorker

            result = self.worker.detect(image, categories, generation_mode="hybrid")
            boxes = LocateAnythingWorker.parse_boxes(result["answer"], w, h)
        except Exception as exc:
            print(f"LocateAnything detect error: {exc}")
            return []

        detections = []
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            cat = categories[i % len(categories)] if categories else "animal"
            cid = class_to_idx.get(cat, -1) if class_to_idx else -1
            det = self._xyxy_to_xywh(x1, y1, x2, y2, w, h)
            det.class_name = cat
            det.class_id = cid
            detections.append(det)
        return detections

    def ground_phrase(self, image: Image.Image, phrase: str) -> List[Detection]:
        if self.worker is None:
            return []

        w, h = image.size
        try:
            from locateanything_worker import LocateAnythingWorker

            result = self.worker.ground_multi(image, phrase, generation_mode="hybrid")
            boxes = LocateAnythingWorker.parse_boxes(result["answer"], w, h)
        except Exception as exc:
            print(f"LocateAnything ground error: {exc}")
            return []

        return [self._xyxy_to_xywh(x1, y1, x2, y2, w, h) for x1, y1, x2, y2 in boxes]


def save_localizer(path: Path, model: BboxLocalizer, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load_localizer(path: Path, device: str = "cpu") -> Tuple[BboxLocalizer, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = BboxLocalizer(pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    return model, ckpt.get("meta", {})

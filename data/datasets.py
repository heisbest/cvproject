"""Dataset loaders for animal-90 and derived datasets."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from .transforms import IMAGENET_MEAN, IMAGENET_STD, get_transforms


class Animal90Dataset(Dataset):
    """Single-animal ImageFolder-style dataset (baseline / animal-90)."""

    def __init__(
        self,
        root: str,
        transform=None,
        full_image_bbox: bool = True,
        indices: list[int] | None = None,
    ):
        self.root = Path(root)
        self.transform = transform
        self.full_image_bbox = full_image_bbox
        self.classes = sorted(
            d.name for d in self.root.iterdir() if d.is_dir()
        )
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        all_samples: List[Tuple[str, int]] = []

        for cls in self.classes:
            cls_dir = self.root / cls
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    all_samples.append((str(cls_dir / img_name), self.class_to_idx[cls]))

        if indices is not None:
            self.samples = [all_samples[i] for i in indices]
        else:
            self.samples = all_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        w, h = image.size

        if self.full_image_bbox:
            bbox = torch.tensor([0.05, 0.05, 0.9, 0.9], dtype=torch.float32)
        else:
            bbox = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "label": label,
            "bbox": bbox,
            "path": path,
            "dataset": "animal-90",
        }


class AnnotatedAnimalDataset(Dataset):
    """
    Dataset with per-image JSON annotations (copy-paste / occlusion).
    Annotation format: {"objects": [{"class": str, "bbox": [x,y,w,h]}]}
    bbox is normalized xywh.
    """

    def __init__(self, root: str, transform=None, target_index: int = 0):
        self.root = Path(root)
        self.transform = transform
        self.target_index = target_index
        self.images_dir = self.root / "images"
        self.ann_dir = self.root / "annotations"

        ann_files = sorted(self.ann_dir.glob("*.json"))
        self.samples = []
        all_classes = set()

        for ann_path in ann_files:
            with open(ann_path, encoding="utf-8") as f:
                ann = json.load(f)
            for obj in ann.get("objects", []):
                all_classes.add(obj["class"])
            self.samples.append(ann_path)

        self.classes = sorted(all_classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        ann_path = self.samples[idx]
        with open(ann_path, encoding="utf-8") as f:
            ann = json.load(f)

        img_path = self.images_dir / ann["file_name"]
        image = Image.open(img_path).convert("RGB")
        objects = ann["objects"]
        obj = objects[self.target_index % len(objects)]

        label = self.class_to_idx[obj["class"]]
        bbox = torch.tensor(obj["bbox"], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "label": label,
            "bbox": bbox,
            "path": str(img_path),
            "dataset": ann.get("dataset", "annotated"),
            "num_objects": len(objects),
        }


def build_class_mapping(*roots: str) -> Tuple[List[str], Dict[str, int]]:
    classes = set()
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        if (root_path / "images").exists():
            ann_dir = root_path / "annotations"
            for ann_path in ann_dir.glob("*.json"):
                with open(ann_path, encoding="utf-8") as f:
                    ann = json.load(f)
                for obj in ann.get("objects", []):
                    classes.add(obj["class"])
        else:
            for d in root_path.iterdir():
                if d.is_dir():
                    classes.add(d.name)

    class_list = sorted(classes)
    return class_list, {c: i for i, c in enumerate(class_list)}


def merge_datasets(datasets: List[Dataset]) -> Dataset:
    from torch.utils.data import ConcatDataset

    return ConcatDataset(datasets)


class LocalizationSampleDataset(Dataset):
    """
    Flatten per-object bbox annotations into (image, bbox) pairs.
    Used to train localizer on multi-animal scenes with tight boxes.
    """

    def __init__(self, root: str, transform=None, min_objects: int = 1):
        self.root = Path(root)
        self.transform = transform
        self.samples: list[dict] = []
        self._load_annotated(min_objects)
        self._load_serengeti(min_objects)

    def _load_annotated(self, min_objects: int) -> None:
        ann_dir = self.root / "annotations"
        img_dir = self.root / "images"
        if not ann_dir.is_dir() or not img_dir.is_dir():
            return

        for ann_path in sorted(ann_dir.glob("*.json")):
            with open(ann_path, encoding="utf-8") as f:
                ann = json.load(f)
            objects = ann.get("objects", [])
            if len(objects) < min_objects:
                continue
            img_path = img_dir / ann["file_name"]
            if not img_path.exists():
                continue
            for obj in objects:
                self.samples.append({
                    "path": str(img_path),
                    "bbox": obj["bbox"],
                    "num_objects": len(objects),
                    "dataset": ann.get("dataset", "annotated"),
                })

    def _load_serengeti(self, min_objects: int) -> None:
        json_path = self._find_serengeti_json()
        if json_path is None:
            return

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        images = {img["id"]: img for img in data.get("images", [])}
        by_image: dict[int, list] = {}
        for ann in data.get("annotations", []):
            if "bbox" not in ann:
                continue
            by_image.setdefault(ann["image_id"], []).append(ann)

        for img_id, anns in by_image.items():
            if len(anns) < min_objects:
                continue
            img_info = images.get(img_id)
            if img_info is None:
                continue
            img_path = self._resolve_serengeti_image(img_info["file_name"])
            if img_path is None:
                continue

            w = img_info.get("width", 1)
            h = img_info.get("height", 1)
            for ann in anns:
                x, y, bw, bh = ann["bbox"]
                self.samples.append({
                    "path": str(img_path),
                    "bbox": [x / w, y / h, bw / w, bh / h],
                    "num_objects": len(anns),
                    "dataset": "serengeti",
                })

    def _find_serengeti_json(self) -> Path | None:
        patterns = [
            "*bounding*box*.json",
            "*bbox*.json",
            "*Bboxes*.json",
            "*multianimal*.json",
            "annotations*.json",
            "snapshot*.json",
        ]
        for pat in patterns:
            hits = sorted(self.root.glob(pat))
            if hits:
                return hits[0]
        meta = self.root / "metadata"
        if meta.is_dir():
            for pat in patterns:
                hits = sorted(meta.glob(pat))
                if hits:
                    return hits[0]
        return None

    def _resolve_serengeti_image(self, file_name: str) -> Path | None:
        candidates = [
            self.root / file_name,
            self.root / "images" / file_name,
            self.root / "images" / Path(file_name).name,
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        image = Image.open(s["path"]).convert("RGB")
        bbox = torch.tensor(s["bbox"], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "label": 0,
            "bbox": bbox,
            "path": s["path"],
            "dataset": s["dataset"],
            "num_objects": s["num_objects"],
        }


class MultiAnimalImageDataset(Dataset):
    """One row per multi-animal image (for localization evaluation)."""

    def __init__(self, root: str, transform=None, min_objects: int = 2):
        self.root = Path(root)
        self.transform = transform
        self.min_objects = min_objects
        flat = LocalizationSampleDataset(root, transform=None, min_objects=min_objects)
        seen: set[str] = set()
        self.samples: list[dict] = []

        for s in flat.samples:
            if s["path"] in seen:
                continue
            seen.add(s["path"])
            bboxes = [x["bbox"] for x in flat.samples if x["path"] == s["path"]]
            self.samples.append({
                "path": s["path"],
                "bboxes": bboxes,
                "num_objects": len(bboxes),
                "dataset": s["dataset"],
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        image = Image.open(s["path"]).convert("RGB")
        primary_bbox = torch.tensor(s["bboxes"][0], dtype=torch.float32)
        all_bboxes = torch.tensor(s["bboxes"], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "label": 0,
            "bbox": primary_bbox,
            "all_bboxes": all_bboxes,
            "path": s["path"],
            "dataset": s["dataset"],
            "num_objects": s["num_objects"],
        }


class SerengetiDataset(Dataset):
    """
    Snapshot Serengeti (LILA) COCO-style bbox JSON + images.

    Expected layout:
      root/
        *.json          # COCO-like: images, annotations, categories
        images/ ...     # or image paths in JSON relative to root
    """

    def __init__(self, root: str, transform=None, metadata_json: str | None = None):
        self.root = Path(root)
        self.transform = transform

        if metadata_json:
            json_path = Path(metadata_json)
        else:
            json_path = self._find_json()
        if json_path is None or not json_path.exists():
            raise FileNotFoundError(
                f"No Serengeti bbox JSON found under {self.root}. "
                "Download from https://lila.science/datasets/snapshot-serengeti/"
            )

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        self.categories = sorted(data.get("categories", []), key=lambda c: c["id"])
        self.class_names = [c["name"] for c in self.categories]
        self.cat_id_to_idx = {c["id"]: i for i, c in enumerate(self.categories)}

        images = {img["id"]: img for img in data.get("images", [])}
        self.samples = []

        for ann in data.get("annotations", []):
            img_id = ann["image_id"]
            if img_id not in images:
                continue
            img_info = images[img_id]
            file_name = img_info["file_name"]
            img_path = self._resolve_image_path(file_name)
            if img_path is None:
                continue

            w = img_info.get("width", 1)
            h = img_info.get("height", 1)
            x, y, bw, bh = ann["bbox"]  # COCO xywh absolute pixels
            bbox = [x / w, y / h, bw / w, bh / h]

            cat_id = ann.get("category_id")
            if cat_id not in self.cat_id_to_idx:
                continue

            self.samples.append({
                "path": str(img_path),
                "label": self.cat_id_to_idx[cat_id],
                "bbox": bbox,
            })

        self.classes = self.class_names

    def _find_json(self) -> Path | None:
        patterns = [
            "*bounding*box*.json",
            "*bbox*.json",
            "annotations*.json",
            "snapshot*.json",
        ]
        for pat in patterns:
            hits = sorted(self.root.glob(pat))
            if hits:
                return hits[0]
        meta = self.root / "metadata"
        if meta.is_dir():
            for pat in patterns:
                hits = sorted(meta.glob(pat))
                if hits:
                    return hits[0]
        return None

    def _resolve_image_path(self, file_name: str) -> Path | None:
        candidates = [
            self.root / file_name,
            self.root / "images" / file_name,
            self.root / "images" / Path(file_name).name,
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        image = Image.open(s["path"]).convert("RGB")
        bbox = torch.tensor(s["bbox"], dtype=torch.float32)
        label = s["label"]

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "label": label,
            "bbox": bbox,
            "path": s["path"],
            "dataset": "serengeti",
        }

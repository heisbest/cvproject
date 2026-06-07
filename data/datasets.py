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
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(split: str = "train", size: int = 224) -> transforms.Compose:
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(int(size * 1.14)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class Animal90Dataset(Dataset):
    """Single-animal ImageFolder-style dataset (baseline / animal-90)."""

    def __init__(self, root: str, transform=None, full_image_bbox: bool = True):
        self.root = Path(root)
        self.transform = transform
        self.full_image_bbox = full_image_bbox
        self.classes = sorted(
            d.name for d in self.root.iterdir() if d.is_dir()
        )
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples: List[Tuple[str, int]] = []

        for cls in self.classes:
            cls_dir = self.root / cls
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    self.samples.append((str(cls_dir / img_name), self.class_to_idx[cls]))

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

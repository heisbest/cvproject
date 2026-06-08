"""Dataset type detection and factory."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

from .datasets import (
    Animal90Dataset,
    LocalizationSampleDataset,
    MultiAnimalImageDataset,
    SerengetiDataset,
    get_transforms,
)


class DatasetType(str, Enum):
    ANIMAL90 = "animal90"
    SERENGETI = "serengeti"
    MULTIANIMAL = "multianimal"
    UNKNOWN = "unknown"


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _is_imagefolder(root: Path) -> bool:
    if not root.is_dir():
        return False
    class_dirs = [d for d in root.iterdir() if d.is_dir()]
    if len(class_dirs) < 2:
        return False
    with_images = 0
    for d in class_dirs[: min(10, len(class_dirs))]:
        if any(p.suffix.lower() in IMG_EXTS for p in d.iterdir() if p.is_file()):
            with_images += 1
    return with_images >= 2


def _find_serengeti_json(root: Path) -> Optional[Path]:
    patterns = [
        "*bounding*box*.json",
        "*bbox*.json",
        "*Bboxes*.json",
        "*multianimal*.json",
        "annotations*.json",
        "snapshot*.json",
    ]
    for pat in patterns:
        hits = sorted(root.glob(pat))
        if hits:
            return hits[0]
    meta = root / "metadata"
    if meta.is_dir():
        for pat in patterns:
            hits = sorted(meta.glob(pat))
            if hits:
                return hits[0]
    return None


def _is_annotated_multianimal(root: Path) -> bool:
    return (root / "annotations").is_dir() and (root / "images").is_dir()


def detect_dataset_type(root: str | Path, forced: Optional[str] = None) -> DatasetType:
    if forced:
        return DatasetType(forced.lower())

    p = Path(root)
    if not p.exists():
        return DatasetType.UNKNOWN

    if _is_annotated_multianimal(p):
        return DatasetType.MULTIANIMAL

    if _find_serengeti_json(p) is not None:
        return DatasetType.SERENGETI

    if _is_imagefolder(p):
        return DatasetType.ANIMAL90

    return DatasetType.UNKNOWN


def build_dataset(
    root: str | Path,
    split: str = "train",
    dataset_type: Optional[str] = None,
    augment_level: str = "standard",
):
    p = Path(root)
    dtype = detect_dataset_type(p, dataset_type)
    transform = get_transforms(split, augment_level=augment_level)

    if dtype == DatasetType.ANIMAL90:
        return Animal90Dataset(str(p), transform=transform)

    if dtype == DatasetType.MULTIANIMAL:
        return AnnotatedAnimalDatasetFromRoot(str(p), transform=transform)

    if dtype == DatasetType.SERENGETI:
        return SerengetiDataset(str(p), transform=transform)

    raise ValueError(
        f"Cannot detect dataset type for {p}. "
        "Use ImageFolder layout (animal-90), annotated multianimal, or Serengeti JSON + images. "
        "Pass --dataset-type animal90|multianimal|serengeti to force."
    )


def AnnotatedAnimalDatasetFromRoot(root: str, transform=None):
    from .datasets import AnnotatedAnimalDataset

    return AnnotatedAnimalDataset(root, transform=transform)


def build_localization_dataset(
    roots: list[str | Path],
    split: str = "train",
    min_objects: int = 1,
):
    """Merge localization samples from multiple roots (multianimal / serengeti)."""
    from torch.utils.data import ConcatDataset

    transform = get_transforms(split)
    parts = []
    for root in roots:
        p = Path(root)
        if not p.exists():
            continue
        ds = LocalizationSampleDataset(str(p), transform=transform, min_objects=min_objects)
        if len(ds) > 0:
            parts.append(ds)

    if not parts:
        raise FileNotFoundError(
            f"No localization samples found under {roots}. "
            "Run scripts/download_localization_data.py or scripts/generate_datasets.py."
        )
    if len(parts) == 1:
        return parts[0]
    return ConcatDataset(parts)


def build_multianimal_eval_dataset(root: str | Path, min_objects: int = 2):
    transform = get_transforms("val")
    return MultiAnimalImageDataset(str(root), transform=transform, min_objects=min_objects)


def collect_class_names(root: str | Path, dataset_type: Optional[str] = None) -> List[str]:
    ds = build_dataset(root, split="val", dataset_type=dataset_type)
    return list(ds.classes)

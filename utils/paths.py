"""Checkpoint / weight directory helpers keyed by dataset path."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


def slugify(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^\w\-]+", "-", s)
    return re.sub(r"-+", "-", s).strip("-") or "dataset"


def dataset_slug(data_path: str | Path) -> str:
    p = Path(data_path).resolve()
    return slugify(p.name)


def weights_dir_for_dataset(
    data_path: str | Path,
    base: str | Path = "./weights",
) -> Path:
    """Weights for a dataset live under weights/<dataset-folder-name>/."""
    return Path(base) / dataset_slug(data_path)


def ensure_weights_dir(data_path: str | Path, base: str | Path = "./weights") -> Path:
    d = weights_dir_for_dataset(data_path, base)
    d.mkdir(parents=True, exist_ok=True)
    return d

"""Stratified train/val splits with configurable random seeds."""

from __future__ import annotations

import random
from typing import Sequence

import torch
from torch.utils.data import Dataset, Subset


def resolve_split_seed(split_seed: str | int | None) -> int:
    """Return an integer seed; 'random' or None picks a new seed each run."""
    if split_seed is None or str(split_seed).lower() == "random":
        return random.randint(0, 2**31 - 1)
    return int(split_seed)


def _labels_from_dataset(dataset: Dataset) -> list[int]:
    labels: list[int] = []
    for i in range(len(dataset)):
        item = dataset[i]
        if isinstance(item, dict):
            labels.append(int(item["label"]))
        else:
            labels.append(int(item[1]))
    return labels


def stratified_index_split(
    labels: list[int],
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[int], list[int], int]:
    """Stratified split returning train/val index lists (no image loading)."""
    by_class: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        by_class.setdefault(label, []).append(idx)

    rng = random.Random(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []

    for indices in by_class.values():
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        n_val = max(int(len(shuffled) * val_ratio), 1 if len(shuffled) > 1 else 0)
        if len(shuffled) <= 1:
            n_val = 0
        val_idx.extend(shuffled[:n_val])
        train_idx.extend(shuffled[n_val:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx, seed


def stratified_train_val_split(
    dataset: Dataset,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[Subset, Subset, int]:
    """
    Stratified split by class label. Returns (train_subset, val_subset, seed_used).
    """
    labels = _labels_from_dataset(dataset)
    by_class: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        by_class.setdefault(label, []).append(idx)

    rng = random.Random(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []

    for indices in by_class.values():
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        n_val = max(int(len(shuffled) * val_ratio), 1 if len(shuffled) > 1 else 0)
        if len(shuffled) <= 1:
            n_val = 0
        val_idx.extend(shuffled[:n_val])
        train_idx.extend(shuffled[n_val:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return Subset(dataset, train_idx), Subset(dataset, val_idx), seed


def random_train_val_split(
    dataset: Dataset,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[Subset, Subset, int]:
    """Fallback random split when labels are unavailable."""
    n = len(dataset)
    n_val = max(int(n * val_ratio), 1)
    n_train = n - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    return train_ds, val_ds, seed

"""Shared evaluation runner."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from data.registry import build_dataset
from models import build_classifier, load_localizer
from utils.metrics import collate, eval_classifier, eval_localizer


def run_evaluation(
    weights_dir: str | Path,
    eval_path: str,
    dataset_type: str | None = None,
    device: torch.device | None = None,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(weights_dir)

    with open(ckpt_dir / "class_names.json", encoding="utf-8") as f:
        class_names = json.load(f)

    dtype = None if dataset_type in (None, "auto") else dataset_type
    eval_ds = build_dataset(eval_path, split="val", dataset_type=dtype)
    n_val = max(len(eval_ds) // 5, 1)
    _, val_ds = random_split(
        eval_ds,
        [len(eval_ds) - n_val, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2, collate_fn=collate)

    loc_ckpt = ckpt_dir / "localizer_best.pth"
    if loc_ckpt.exists():
        loc_model, meta = load_localizer(loc_ckpt, device=str(device))
        iou, n = eval_localizer(loc_model, loader, device)
        print(f"[Localization] IoU={iou:.4f}  n={n}  meta={meta}")
    else:
        print("[Localization] skipped (no localizer_best.pth)")

    cls_ckpt = ckpt_dir / "classifier_cbam_best.pth"
    if cls_ckpt.exists():
        ckpt = torch.load(cls_ckpt, map_location=device, weights_only=False)
        model = build_classifier(len(class_names), pretrained=False)
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)

        acc_masked, n = eval_classifier(model, loader, device, use_mask=True)
        acc_full, _ = eval_classifier(model, loader, device, use_mask=False)
        print(f"[Classification] acc(masked)={acc_masked:.4f}  acc(full)={acc_full:.4f}  n={n}")
    else:
        print("[Classification] skipped (no classifier_cbam_best.pth)")

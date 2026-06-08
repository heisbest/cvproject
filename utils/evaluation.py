"""Shared evaluation runner."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.registry import build_dataset, build_multianimal_eval_dataset
from data.splits import resolve_split_seed, stratified_train_val_split
from models import load_localizer
from models.ensemble import load_ensemble_from_dir
from utils.metrics import (
    collate,
    eval_classifier,
    eval_ensemble,
    eval_individual_backbones,
    eval_localizer,
    eval_localizer_multianimal,
)


def _make_eval_loader(eval_path, dataset_type, split_seed, batch_size=32):
    dtype = None if dataset_type in (None, "auto") else dataset_type
    eval_ds = build_dataset(eval_path, split="val", dataset_type=dtype)
    seed = resolve_split_seed(split_seed)
    _, val_ds, _ = stratified_train_val_split(eval_ds, val_ratio=0.2, seed=seed)
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collate)
    return loader, seed


def run_evaluation(
    weights_dir: str | Path,
    eval_path: str,
    dataset_type: str | None = None,
    device: torch.device | None = None,
    split_seed: str | int | None = 42,
    loc_eval_path: str | None = None,
    classifier_backbones: tuple[str, ...] | None = None,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(weights_dir)

    with open(ckpt_dir / "class_names.json", encoding="utf-8") as f:
        class_names = json.load(f)

    loader, seed = _make_eval_loader(eval_path, dataset_type, split_seed)
    print(f"[Eval split] seed={seed}  n={len(loader.dataset)}")

    loc_ckpt = ckpt_dir / "localizer_best.pth"
    if loc_ckpt.exists():
        loc_model, meta = load_localizer(loc_ckpt, device=str(device))

        iou, n = eval_localizer(loc_model, loader, device)
        print(f"[Localization @ {eval_path}] IoU={iou:.4f}  n={n}  (single-animal bbox baseline)")

        loc_eval = loc_eval_path or str(Path(eval_path).parent / "animal-90-multianimal")
        if Path(loc_eval).exists():
            multi_ds = build_multianimal_eval_dataset(loc_eval, min_objects=2)
            multi_loader = DataLoader(
                multi_ds, batch_size=16, shuffle=False, num_workers=2, collate_fn=collate
            )
            best_iou, primary_iou, mn = eval_localizer_multianimal(loc_model, multi_loader, device)
            print(
                f"[Localization @ {loc_eval}] best-of-GT IoU={best_iou:.4f}  "
                f"primary IoU={primary_iou:.4f}  n={mn}  (multi-animal)"
            )
    else:
        print("[Localization] skipped (no localizer_best.pth)")

    ensemble = load_ensemble_from_dir(
        ckpt_dir, len(class_names), device=device, backbones=classifier_backbones
    )
    if ensemble is not None:
        individual = eval_individual_backbones(ensemble, loader, device, use_mask=True)
        for name, acc in individual.items():
            print(f"[Classification/{name}] acc(masked)={acc:.4f}")

        acc_masked, n = eval_ensemble(ensemble, loader, device, use_mask=True)
        acc_full, _ = eval_ensemble(ensemble, loader, device, use_mask=False)
        print(
            f"[Classification/ensemble] acc(masked)={acc_masked:.4f}  "
            f"acc(full)={acc_full:.4f}  n={n}  (uncertainty fusion)"
        )
    else:
        from models import build_classifier

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
            print("[Classification] skipped (no ensemble or legacy checkpoint)")

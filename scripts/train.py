"""
Train localizer + classifier ensemble on a user-specified dataset.

Weights are saved under weights/<dataset-folder-name>/.
Use --resume to continue from existing checkpoints in that folder.
Each run uses a new train/val split unless --split-seed is fixed.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.datasets import Animal90Dataset, get_transforms
from data.registry import DatasetType, build_dataset, build_localization_dataset, collect_class_names, detect_dataset_type
from data.splits import random_train_val_split, resolve_split_seed, stratified_index_split, stratified_train_val_split
from models import (
    BboxLocalizer,
    apply_mask_to_image,
    localization_loss,
    save_localizer,
)
from models.ensemble import (
    CHECKPOINT_NAMES,
    CLASSIFIER_BACKBONES,
    build_classifier_backbone,
)
from utils.metrics import collate, eval_classifier, eval_localizer
from utils.paths import ensure_weights_dir, weights_dir_for_dataset


def _make_loaders(data_path, dtype, batch_size, val_ratio, split_seed, augment_level="strong"):
    path = Path(data_path)
    if detect_dataset_type(path, dtype) == DatasetType.ANIMAL90:
        catalog = Animal90Dataset(str(path), transform=None)
        labels = [lab for _, lab in catalog.samples]
        seed = resolve_split_seed(split_seed)
        train_idx, val_idx, seed = stratified_index_split(labels, val_ratio=val_ratio, seed=seed)
        train_ds = Animal90Dataset(
            str(path),
            transform=get_transforms("train", augment_level=augment_level),
            indices=train_idx,
        )
        val_ds = Animal90Dataset(
            str(path),
            transform=get_transforms("val"),
            indices=val_idx,
        )
    else:
        full_train = build_dataset(
            path, split="train", dataset_type=dtype, augment_level=augment_level
        )
        seed = resolve_split_seed(split_seed)
        try:
            train_ds, val_ds, seed = stratified_train_val_split(
                full_train, val_ratio=val_ratio, seed=seed
            )
        except Exception:
            train_ds, val_ds, seed = random_train_val_split(
                full_train, val_ratio=val_ratio, seed=seed
            )

    print(f"Train/val split seed={seed}  train={len(train_ds)}  val={len(val_ds)}  aug={augment_level}")
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4,
        pin_memory=True, collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4,
        pin_memory=True, collate_fn=collate,
    )
    return train_loader, val_loader, seed


def _make_loc_loaders(loc_paths, batch_size, val_ratio, split_seed):
    full = build_localization_dataset(loc_paths, split="train", min_objects=1)
    seed = resolve_split_seed(split_seed)
    train_ds, val_ds, seed = random_train_val_split(full, val_ratio=val_ratio, seed=seed)
    print(f"Localization split seed={seed}  train={len(train_ds)}  val={len(val_ds)}")
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collate
    )
    return train_loader, val_loader, seed


def _resolve_loc_paths(args, data_path: Path) -> list[str]:
    if args.loc_data_paths:
        return [p for p in args.loc_data_paths if Path(p).exists()]

    defaults = [
        data_path.parent / "animal-90-multianimal",
        data_path.parent / "serengeti-multianimal",
    ]
    existing = [str(p) for p in defaults if p.exists()]
    if existing:
        return existing

    multianimal = data_path.parent / "animal-90-multianimal"
    if not multianimal.exists() and data_path.exists():
        print("Generating animal-90-multianimal for localization training...")
        import subprocess
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "generate_datasets.py"),
                "--source",
                str(data_path),
                "--out-multianimal",
                str(multianimal),
            ],
            check=False,
        )
        if multianimal.exists():
            return [str(multianimal)]
    return []


def train_localizer(args, device, train_loader, val_loader, out_dir: Path):
    model = BboxLocalizer(pretrained=not args.resume).to(device)

    start_epoch = 0
    best_iou = 0.0
    loc_ckpt = out_dir / "localizer_best.pth"

    if args.resume and loc_ckpt.exists():
        ckpt = torch.load(loc_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        best_iou = ckpt.get("meta", {}).get("val_iou", 0.0)
        start_epoch = ckpt.get("meta", {}).get("epoch", 0)
        print(f"Resumed localizer from {loc_ckpt} (best_iou={best_iou:.4f})")

    optimizer = optim.Adam(model.parameters(), lr=args.loc_lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

    for epoch in range(start_epoch, args.loc_epochs):
        model.train()
        running = 0.0
        for batch in tqdm(train_loader, desc=f"Loc E{epoch+1}/{args.loc_epochs}"):
            images = batch["image"].to(device)
            targets = batch["bbox"].to(device)
            optimizer.zero_grad()
            preds = model(images)
            loss = localization_loss(preds, targets)
            loss.backward()
            optimizer.step()
            running += loss.item()

        scheduler.step()
        val_iou, _ = eval_localizer(model, val_loader, device)
        print(f"[Localizer] epoch {epoch+1} loss={running/len(train_loader):.4f} val_iou={val_iou:.4f}")

        if val_iou >= best_iou:
            best_iou = val_iou
            save_localizer(loc_ckpt, model, {"val_iou": val_iou, "epoch": epoch + 1})

    print(f"Localizer best IoU: {best_iou:.4f}")
    return model


def _is_head_param(name: str, backbone_name: str) -> bool:
    if backbone_name == "resnet_cbam":
        return name.startswith("fc.")
    if backbone_name in ("efficientnet_b3", "convnext_t"):
        return name.startswith("backbone.classifier.")
    return name.startswith("fc.") or ".classifier." in name


def _is_warmup_trainable(name: str, backbone_name: str) -> bool:
    """Parameters trainable before full backbone unfreeze."""
    if backbone_name == "resnet_cbam":
        return name.startswith("fc.") or name.startswith("layer4.")
    if backbone_name in ("efficientnet_b3", "convnext_t"):
        return name.startswith("backbone.classifier.")
    return _is_head_param(name, backbone_name)


def _param_groups(model: nn.Module, backbone_name: str, lr: float, weight_decay: float):
    head_params, body_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if _is_head_param(name, backbone_name):
            head_params.append(param)
        else:
            body_params.append(param)
    groups = []
    if body_params:
        groups.append({"params": body_params, "lr": lr * 0.25, "weight_decay": weight_decay})
    if head_params:
        groups.append({"params": head_params, "lr": lr, "weight_decay": weight_decay * 0.5})
    if not groups:
        groups.append({"params": [p for p in model.parameters() if p.requires_grad], "lr": lr})
    return groups


def _set_backbone_trainable(model: nn.Module, backbone_name: str, trainable: bool) -> None:
    for name, param in model.named_parameters():
        if trainable:
            param.requires_grad = True
        else:
            param.requires_grad = _is_warmup_trainable(name, backbone_name)


def train_single_classifier(
    backbone_name: str,
    args,
    device,
    train_loader,
    val_loader,
    out_dir: Path,
    class_names: list[str],
):
    num_classes = len(class_names)
    model = build_classifier_backbone(backbone_name, num_classes, pretrained=not args.resume).to(device)

    ckpt_name = CHECKPOINT_NAMES[backbone_name]
    cls_ckpt = out_dir / ckpt_name
    best_acc = 0.0
    start_epoch = 0

    if args.resume and cls_ckpt.exists():
        ckpt = torch.load(cls_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        best_acc = ckpt.get("val_acc", 0.0)
        start_epoch = ckpt.get("epoch", 0)
        print(f"Resumed {backbone_name} from {cls_ckpt} (best_acc={best_acc:.4f})")
    else:
        _set_backbone_trainable(model, backbone_name, trainable=False)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(
        _param_groups(model, backbone_name, args.cls_lr, args.weight_decay),
        betas=(0.9, 0.999),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.cls_epochs, 1),
        eta_min=args.cls_lr * args.min_lr_ratio,
    )
    if start_epoch > 0:
        for _ in range(start_epoch):
            scheduler.step()

    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    use_mask = not args.cls_no_mask

    for epoch in range(start_epoch, args.cls_epochs):
        if epoch == args.unfreeze_epoch and not args.resume:
            _set_backbone_trainable(model, backbone_name, trainable=True)
            optimizer = optim.AdamW(
                _param_groups(model, backbone_name, args.cls_lr, args.weight_decay),
                betas=(0.9, 0.999),
            )
            remaining = max(args.cls_epochs - epoch, 1)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=remaining,
                eta_min=args.cls_lr * args.min_lr_ratio,
            )
            print(f"[{backbone_name}] unfroze backbone at epoch {epoch + 1}")

        model.train()
        running = 0.0
        correct = 0
        total = 0

        for batch in tqdm(train_loader, desc=f"{backbone_name} E{epoch+1}/{args.cls_epochs}"):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            bboxes = batch["bbox"].to(device, non_blocking=True)
            inputs = apply_mask_to_image(images, bboxes) if use_mask else images

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_acc = correct / max(total, 1)
        val_acc, _ = eval_classifier(model, val_loader, device, use_mask=use_mask)
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"[{backbone_name}] epoch {epoch+1} loss={running/len(train_loader):.4f} "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} lr={lr_now:.2e}"
        )

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "class_names": class_names,
                    "val_acc": val_acc,
                    "epoch": epoch + 1,
                    "backbone": backbone_name,
                },
                cls_ckpt,
            )

    print(f"[{backbone_name}] best acc: {best_acc:.4f}")
    return model


def train_classifier_ensemble(args, device, train_loader, val_loader, out_dir: Path, class_names: list[str]):
    backbones = tuple(b for b in CLASSIFIER_BACKBONES if b in args.backbones)
    print(f"Training classifier backbones: {backbones}")
    for name in backbones:
        train_single_classifier(name, args, device, train_loader, val_loader, out_dir, class_names)


from utils.evaluation import run_evaluation


def main():
    parser = argparse.ArgumentParser(description="Train animal recognition models")
    parser.add_argument("--data-path", required=True, help="Training dataset root path")
    parser.add_argument(
        "--dataset-type",
        choices=["animal90", "multianimal", "serengeti", "auto"],
        default="auto",
    )
    parser.add_argument("--weights-dir", default=None, help="Override weights dir (default: weights/<name>)")
    parser.add_argument("--eval-path", default=None, help="Evaluation set path (eval-only or post-train eval)")
    parser.add_argument(
        "--loc-data-paths",
        nargs="*",
        default=None,
        help="Multi-animal localization datasets (default: multianimal + serengeti subset)",
    )
    parser.add_argument(
        "--loc-eval-path",
        default=None,
        help="Multi-animal eval path for localization metrics",
    )
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate, do not train")
    parser.add_argument("--resume", action="store_true", help="Continue from existing weights in weights dir")
    parser.add_argument(
        "--split-seed",
        default="random",
        help="Train/val split seed; 'random' picks a new seed every run (even with --resume)",
    )
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--loc-epochs", type=int, default=20)
    parser.add_argument("--cls-epochs", type=int, default=50)
    parser.add_argument("--loc-lr", type=float, default=1e-3)
    parser.add_argument("--cls-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--min-lr-ratio", type=float, default=0.01, help="Cosine eta_min = cls_lr * ratio")
    parser.add_argument("--unfreeze-epoch", type=int, default=5, help="Epoch to unfreeze full backbone")
    parser.add_argument(
        "--augment-level",
        choices=["none", "standard", "strong"],
        default="strong",
        help="Training augmentation strength (strong recommended for Animal-90)",
    )
    parser.add_argument(
        "--cls-use-mask",
        action="store_true",
        help="Apply bbox mask during classifier training (default: full image for Animal-90 accuracy)",
    )
    parser.add_argument("--amp", action="store_true", default=True, help="Mixed precision training")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--backbones",
        nargs="*",
        default=list(CLASSIFIER_BACKBONES),
        choices=list(CLASSIFIER_BACKBONES),
        help="Classifier backbones to train",
    )
    parser.add_argument("--train-only", choices=["resnet", "maskformer", "all"], default=None,
                        help="Train only one model stage: resnet=classifier, maskformer=localizer, all=both")
    parser.add_argument("--no-loc", action="store_true", help="Skip localizer training")
    parser.add_argument("--no-cls", action="store_true", help="Skip classifier training")
    args = parser.parse_args()

    if args.train_only is not None:
        args.no_loc = args.train_only == "resnet"
        args.no_cls = args.train_only == "maskformer"

    if args.cls_use_mask:
        args.cls_no_mask = False
    else:
        args.cls_no_mask = True
    if args.no_amp:
        args.amp = False

    dtype = None if args.dataset_type == "auto" else args.dataset_type
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    out_dir = Path(args.weights_dir) if args.weights_dir else ensure_weights_dir(data_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detected = detect_dataset_type(data_path, dtype)
    print(f"Device: {device}")
    print(f"Dataset: {data_path}  type={detected.value}")
    print(f"Weights: {out_dir}")

    class_names = collect_class_names(data_path, dtype)
    with open(out_dir / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)
    print(f"Classes ({len(class_names)}): {class_names[:8]}...")

    if args.eval_only:
        eval_path = args.eval_path or str(data_path)
        print("\n=== Eval only ===")
        run_evaluation(
            weights_dir=out_dir,
            eval_path=eval_path,
            dataset_type=args.dataset_type,
            device=device,
            split_seed=args.split_seed,
            loc_eval_path=args.loc_eval_path,
        )
        return

    train_loader, val_loader, cls_split_seed = _make_loaders(
        data_path, dtype, args.batch_size, args.val_ratio, args.split_seed, args.augment_level
    )

    loc_split_seed = None
    if not args.no_loc:
        loc_paths = _resolve_loc_paths(args, data_path)
        if not loc_paths:
            print("\n=== Stage 1: Localization skipped (no multi-animal data) ===")
            print("Run: python scripts/download_localization_data.py")
        else:
            print(f"\n=== Stage 1: Localization on {loc_paths} ===")
            loc_train, loc_val, loc_split_seed = _make_loc_loaders(
                loc_paths, args.batch_size, args.val_ratio, args.split_seed
            )
            train_localizer(args, device, loc_train, loc_val, out_dir)

    if not args.no_cls:
        print("\n=== Stage 2: Multi-backbone classification + uncertainty fusion ===")
        train_classifier_ensemble(args, device, train_loader, val_loader, out_dir, class_names)

    eval_path = args.eval_path or str(data_path)
    loc_eval = args.loc_eval_path or str(data_path.parent / "animal-90-multianimal")
    print("\n=== Post-train evaluation ===")
    run_evaluation(
        weights_dir=out_dir,
        eval_path=eval_path,
        dataset_type=args.dataset_type,
        device=device,
        split_seed=args.split_seed,
        loc_eval_path=loc_eval,
    )

    state = {
        "data_path": str(data_path.resolve()),
        "dataset_type": detected.value,
        "loc_epochs": args.loc_epochs,
        "cls_epochs": args.cls_epochs,
        "resume": args.resume,
        "split_seed": cls_split_seed,
        "loc_split_seed": loc_split_seed,
        "backbones": list(args.backbones),
        "val_ratio": args.val_ratio,
    }
    with open(out_dir / "train_state.json", "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    print("\nTraining complete.")


if __name__ == "__main__":
    main()

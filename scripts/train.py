"""
Train localizer + classifier on a user-specified dataset.

Weights are saved under weights/<dataset-folder-name>/.
Use --resume to continue from existing checkpoints in that folder.
Use --eval-only with --eval-path to skip training and only evaluate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.registry import DatasetType, build_dataset, collect_class_names, detect_dataset_type
from models import (
    BboxLocalizer,
    apply_mask_to_image,
    build_classifier,
    localization_loss,
    save_localizer,
)
from utils.metrics import collate, eval_classifier, eval_localizer
from utils.paths import ensure_weights_dir, weights_dir_for_dataset


def _make_loaders(data_path, dtype, batch_size, val_ratio=0.15):
    full_train = build_dataset(data_path, split="train", dataset_type=dtype)
    n_val = max(int(len(full_train) * val_ratio), 1)
    n_train = len(full_train) - n_val
    train_ds, val_ds = random_split(
        full_train, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collate
    )
    return train_loader, val_loader


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


def train_classifier(args, device, train_loader, val_loader, out_dir: Path, class_names: list[str]):
    num_classes = len(class_names)
    model = build_classifier(num_classes, pretrained=not args.resume).to(device)

    cls_ckpt = out_dir / "classifier_cbam_best.pth"
    best_acc = 0.0
    start_epoch = 0

    if args.resume and cls_ckpt.exists():
        ckpt = torch.load(cls_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        best_acc = ckpt.get("val_acc", 0.0)
        start_epoch = ckpt.get("epoch", 0)
        print(f"Resumed classifier from {cls_ckpt} (best_acc={best_acc:.4f})")
    else:
        for name, param in model.named_parameters():
            if not name.startswith("fc") and not name.startswith("layer4"):
                param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.cls_lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in range(start_epoch, args.cls_epochs):
        model.train()
        running = 0.0
        correct = 0
        total = 0

        for batch in tqdm(train_loader, desc=f"Cls E{epoch+1}/{args.cls_epochs}"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            bboxes = batch["bbox"].to(device)
            masked = apply_mask_to_image(images, bboxes)

            optimizer.zero_grad()
            outputs = model(masked)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_acc = correct / max(total, 1)
        val_acc, _ = eval_classifier(model, val_loader, device, use_mask=True)
        print(f"[Classifier] epoch {epoch+1} loss={running/len(train_loader):.4f} "
              f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save({
                "state_dict": model.state_dict(),
                "class_names": class_names,
                "val_acc": val_acc,
                "epoch": epoch + 1,
            }, cls_ckpt)

    print(f"Classifier best acc: {best_acc:.4f}")
    return model


from utils.evaluation import run_evaluation


def main():
    parser = argparse.ArgumentParser(description="Train animal recognition models")
    parser.add_argument("--data-path", required=True, help="Training dataset root path")
    parser.add_argument("--dataset-type", choices=["animal90", "serengeti", "auto"], default="auto")
    parser.add_argument("--weights-dir", default=None, help="Override weights dir (default: weights/<name>)")
    parser.add_argument("--eval-path", default=None, help="Evaluation set path (eval-only or post-train eval)")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate, do not train")
    parser.add_argument("--resume", action="store_true", help="Continue from existing weights in weights dir")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--loc-epochs", type=int, default=20)
    parser.add_argument("--cls-epochs", type=int, default=25)
    parser.add_argument("--loc-lr", type=float, default=1e-3)
    parser.add_argument("--cls-lr", type=float, default=1e-3)
    parser.add_argument("--no-loc", action="store_true", help="Skip localizer training")
    parser.add_argument("--no-cls", action="store_true", help="Skip classifier training")
    args = parser.parse_args()

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
        )
        return

    train_loader, val_loader = _make_loaders(data_path, dtype, args.batch_size)

    if not args.no_loc:
        print("\n=== Stage 1: Localization (separate backprop) ===")
        train_localizer(args, device, train_loader, val_loader, out_dir)

    if not args.no_cls:
        print("\n=== Stage 2: Classification CBAM + mask (separate backprop) ===")
        train_classifier(args, device, train_loader, val_loader, out_dir, class_names)

    eval_path = args.eval_path or str(data_path)
    print("\n=== Post-train evaluation ===")
    run_evaluation(
        weights_dir=out_dir,
        eval_path=eval_path,
        dataset_type=args.dataset_type,
        device=device,
    )

    state = {
        "data_path": str(data_path.resolve()),
        "dataset_type": detected.value,
        "loc_epochs": args.loc_epochs,
        "cls_epochs": args.cls_epochs,
        "resume": args.resume,
    }
    with open(out_dir / "train_state.json", "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    print("\nTraining complete.")


if __name__ == "__main__":
    main()

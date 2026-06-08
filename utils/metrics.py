"""Shared metrics and collate helpers."""

import torch

from models import apply_mask_to_image
from models.ensemble import UncertaintyFusionEnsemble
from models.localizer import box_iou


def collate(batch):
    out = {
        "image": torch.stack([b["image"] for b in batch]),
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "bbox": torch.stack([b["bbox"] for b in batch]),
    }
    if "all_bboxes" in batch[0]:
        out["all_bboxes"] = [b["all_bboxes"] for b in batch]
    if "num_objects" in batch[0]:
        out["num_objects"] = torch.tensor([b["num_objects"] for b in batch], dtype=torch.long)
    return out


@torch.no_grad()
def eval_localizer(model, loader, device) -> tuple[float, float]:
    model.eval()
    total_iou = 0.0
    n = 0
    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["bbox"].to(device)
        preds = model(images)
        iou = box_iou(preds, targets).diag()
        total_iou += iou.sum().item()
        n += iou.numel()
    return total_iou / max(n, 1), n


@torch.no_grad()
def eval_localizer_multianimal(model, loader, device) -> tuple[float, float, float]:
    """
    Evaluate on multi-animal images: best IoU against any GT box + mean IoU vs primary.
    """
    model.eval()
    best_iou_sum = 0.0
    primary_iou_sum = 0.0
    n = 0

    for batch in loader:
        images = batch["image"].to(device)
        preds = model(images)
        all_bboxes = batch.get("all_bboxes")

        if all_bboxes is None:
            targets = batch["bbox"].to(device)
            iou = box_iou(preds, targets).diag()
            best_iou_sum += iou.sum().item()
            primary_iou_sum += iou.sum().item()
            n += iou.numel()
            continue

        for i in range(preds.size(0)):
            gt = all_bboxes[i].to(device)
            pred = preds[i : i + 1]
            ious = box_iou(pred.expand(gt.size(0), -1), gt)
            best_iou_sum += ious.max().item()
            primary_iou_sum += box_iou(pred, gt[0:1]).item()
            n += 1

    denom = max(n, 1)
    return best_iou_sum / denom, primary_iou_sum / denom, n


@torch.no_grad()
def eval_classifier(model, loader, device, use_mask: bool = True) -> tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        bboxes = batch["bbox"].to(device)

        if use_mask:
            images = apply_mask_to_image(images, bboxes)

        outputs = model(images)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1), total


@torch.no_grad()
def eval_ensemble(ensemble: UncertaintyFusionEnsemble, loader, device, use_mask: bool = True) -> tuple[float, float]:
    ensemble.eval()
    correct = 0
    total = 0

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        bboxes = batch["bbox"].to(device)

        if use_mask:
            images = apply_mask_to_image(images, bboxes)

        outputs = ensemble(images)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / max(total, 1), total


@torch.no_grad()
def eval_individual_backbones(ensemble: UncertaintyFusionEnsemble, loader, device, use_mask: bool = True) -> dict:
    ensemble.eval()
    stats = {name: {"correct": 0, "total": 0} for name in ensemble.backbone_names}

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        bboxes = batch["bbox"].to(device)

        if use_mask:
            images = apply_mask_to_image(images, bboxes)

        for model, name in zip(ensemble.models, ensemble.backbone_names):
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            stats[name]["correct"] += (preds == labels).sum().item()
            stats[name]["total"] += labels.size(0)

    return {
        name: s["correct"] / max(s["total"], 1)
        for name, s in stats.items()
    }

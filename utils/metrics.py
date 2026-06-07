"""Shared metrics and collate helpers."""

import torch

from models import apply_mask_to_image
from models.localizer import box_iou


def collate(batch):
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "bbox": torch.stack([b["bbox"] for b in batch]),
    }


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

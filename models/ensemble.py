"""
Multi-backbone ensemble with uncertainty-aware softmax fusion.

Each backbone produces logits -> temperature-scaled softmax -> entropy-based
confidence weights. Fused probabilities are a weighted sum (not a plain average).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .convnext_classifier import build_convnext_classifier
from .efficientnet_classifier import build_efficientnet_classifier
from .resnet_cbam import build_classifier as build_resnet_classifier

CLASSIFIER_BACKBONES = ("resnet_cbam", "efficientnet_b3", "convnext_t")

BACKBONE_BUILDERS = {
    "resnet_cbam": build_resnet_classifier,
    "efficientnet_b3": build_efficientnet_classifier,
    "convnext_t": build_convnext_classifier,
}

CHECKPOINT_NAMES = {
    "resnet_cbam": "classifier_resnet_cbam_best.pth",
    "efficientnet_b3": "classifier_efficientnet_b3_best.pth",
    "convnext_t": "classifier_convnext_t_best.pth",
}

LEGACY_RESNET_CKPT = "classifier_cbam_best.pth"


def build_classifier_backbone(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if name not in BACKBONE_BUILDERS:
        raise ValueError(f"Unknown backbone {name!r}, expected one of {CLASSIFIER_BACKBONES}")
    return BACKBONE_BUILDERS[name](num_classes, pretrained=pretrained)


def uncertainty_fusion(
    logits_list: list[torch.Tensor],
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fuse multiple logits with entropy-based confidence weighting.

    Returns (fused_logits, model_weights) where model_weights shape is (B, M).
    """
    if len(logits_list) == 1:
        return logits_list[0], torch.ones(logits_list[0].size(0), 1, device=logits_list[0].device)

    probs_list = [F.softmax(logits / temperature, dim=1) for logits in logits_list]
    entropies = torch.stack(
        [-(p * (p.clamp(min=1e-8)).log()).sum(dim=1) for p in probs_list],
        dim=1,
    )
    max_probs = torch.stack([p.max(dim=1).values for p in probs_list], dim=1)

    # Lower entropy and higher peak probability -> higher weight.
    confidence = torch.exp(-entropies) * max_probs
    weights = confidence / confidence.sum(dim=1, keepdim=True).clamp(min=1e-8)

    probs_stack = torch.stack(probs_list, dim=1)
    fused_probs = (probs_stack * weights.unsqueeze(-1)).sum(dim=1)
    fused_logits = torch.log(fused_probs.clamp(min=1e-8))
    return fused_logits, weights


class UncertaintyFusionEnsemble(nn.Module):
    """Wraps multiple classifiers and fuses their softmax outputs by uncertainty."""

    def __init__(
        self,
        models: Iterable[nn.Module],
        temperature: float = 1.0,
        backbone_names: list[str] | None = None,
    ):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.temperature = temperature
        self.backbone_names = backbone_names or []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits_list = [model(x) for model in self.models]
        fused_logits, _ = uncertainty_fusion(logits_list, temperature=self.temperature)
        return fused_logits

    def forward_with_details(self, x: torch.Tensor) -> dict:
        logits_list = [model(x) for model in self.models]
        fused_logits, weights = uncertainty_fusion(logits_list, temperature=self.temperature)
        return {
            "logits": fused_logits,
            "individual_logits": logits_list,
            "fusion_weights": weights,
        }


def load_ensemble_from_dir(
    weights_dir: str | Path,
    num_classes: int,
    device: torch.device | str = "cpu",
    backbones: tuple[str, ...] | None = None,
) -> UncertaintyFusionEnsemble | None:
    """Load all available backbone checkpoints and build an ensemble."""
    if backbones is None:
        backbones = CLASSIFIER_BACKBONES

    ckpt_dir = Path(weights_dir)
    models: list[nn.Module] = []
    loaded_names: list[str] = []

    for name in backbones:
        ckpt_path = ckpt_dir / CHECKPOINT_NAMES[name]
        if not ckpt_path.exists() and name == "resnet_cbam":
            legacy = ckpt_dir / LEGACY_RESNET_CKPT
            if legacy.exists():
                ckpt_path = legacy

        if not ckpt_path.exists():
            continue

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = build_classifier_backbone(name, num_classes, pretrained=False)
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        model.eval()
        models.append(model)
        loaded_names.append(name)

    if not models:
        return None

    ensemble = UncertaintyFusionEnsemble(models, backbone_names=loaded_names)
    ensemble.to(device)
    ensemble.eval()
    return ensemble

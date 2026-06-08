"""ConvNeXt-Tiny classifier with ImageNet pretrained weights."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny


class ConvNeXtTClassifier(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = convnext_tiny(weights=weights)
        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier = nn.Sequential(
            self.backbone.classifier[0],
            self.backbone.classifier[1],
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


def build_convnext_classifier(num_classes: int, pretrained: bool = True) -> ConvNeXtTClassifier:
    return ConvNeXtTClassifier(num_classes=num_classes, pretrained=pretrained)

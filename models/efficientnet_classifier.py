"""EfficientNet-B3 classifier with ImageNet pretrained weights."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import EfficientNet_B3_Weights, efficientnet_b3


class EfficientNetB3Classifier(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = efficientnet_b3(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=False),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


def build_efficientnet_classifier(num_classes: int, pretrained: bool = True) -> EfficientNetB3Classifier:
    return EfficientNetB3Classifier(num_classes=num_classes, pretrained=pretrained)

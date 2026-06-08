"""ResNet50 with CBAM inserted into each Bottleneck block."""

import torch
import torch.nn as nn
from torchvision.models.resnet import Bottleneck, ResNet

from .cbam import CBAM


class BottleneckCBAM(Bottleneck):
    """Bottleneck with CBAM applied after conv3, before residual add."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cbam = CBAM(self.conv3.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.cbam(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


class ResNet50CBAM(ResNet):
    def __init__(self, num_classes: int = 1000, pretrained: bool = True):
        super().__init__(block=BottleneckCBAM, layers=[3, 4, 6, 3], num_classes=num_classes)

        if pretrained:
            from torchvision.models import ResNet50_Weights, resnet50

            state = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).state_dict()
            if num_classes != 1000:
                state = {k: v for k, v in state.items() if not k.startswith("fc.")}
            missing, unexpected = self.load_state_dict(state, strict=False)
            cbam_keys = [k for k in missing if "cbam" in k]
            other_missing = [k for k in missing if "cbam" not in k]
            if num_classes != 1000:
                other_missing = [k for k in other_missing if not k.startswith("fc.")]
            if other_missing:
                print(f"Warning: non-CBAM keys not loaded: {other_missing[:5]}...")


def build_classifier(num_classes: int, pretrained: bool = True) -> ResNet50CBAM:
    model = ResNet50CBAM(num_classes=num_classes, pretrained=pretrained)
    return model


def apply_mask_to_image(image: torch.Tensor, bbox: torch.Tensor) -> torch.Tensor:
    """
    Apply a soft rectangular mask from normalized xywh bbox.
    image: (B, 3, H, W), bbox: (B, 4) in [x, y, w, h] normalized 0-1.
    """
    b, _, h, w = image.shape
    mask = torch.zeros(b, 1, h, w, device=image.device, dtype=image.dtype)

    for i in range(b):
        x, y, bw, bh = bbox[i]
        x1 = int(max(0, x.item() * w))
        y1 = int(max(0, y.item() * h))
        x2 = int(min(w, (x + bw).item() * w))
        y2 = int(min(h, (y + bh).item() * h))
        if x2 > x1 and y2 > y1:
            mask[i, 0, y1:y2, x1:x2] = 1.0

    return image * mask

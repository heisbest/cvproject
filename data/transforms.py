"""Train/val image transforms with configurable augmentation strength."""

from __future__ import annotations

from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(
    split: str = "train",
    size: int = 224,
    augment_level: str = "standard",
):
    """
    augment_level:
      - none: resize + center crop only
      - standard: legacy baseline (flip + mild color jitter)
      - strong: geometric + photometric + random erasing (for Animal-90 accuracy)
    """
    if split != "train":
        return transforms.Compose([
            transforms.Resize(int(size * 1.14)),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    if augment_level == "none":
        return transforms.Compose([
            transforms.Resize(int(size * 1.14)),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    if augment_level == "standard":
        return transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    # strong: geometry + rotation + photometric (incl. saturation/chroma) + erasing
    aug = [
        transforms.RandomResizedCrop(size, scale=(0.65, 1.0), ratio=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(degrees=25),
        transforms.RandomAffine(
            degrees=10,
            translate=(0.1, 0.1),
            scale=(0.88, 1.12),
            shear=12,
        ),
        transforms.ColorJitter(
            brightness=0.35,
            contrast=0.35,
            saturation=0.35,
            hue=0.06,
        ),
        transforms.RandomGrayscale(p=0.08),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2)),
    ]
    try:
        aug.insert(-1, transforms.TrivialAugmentWide())
    except AttributeError:
        pass

    aug.extend([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
    ])
    return transforms.Compose(aug)

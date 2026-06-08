"""
Download ImageNet pretrained backbone weights for offline / proxy training.

Caches under ~/.cache/torch/hub/checkpoints/ (same path torchvision uses).

Usage:
  export http_proxy="http://127.0.0.1:7893"
  export https_proxy="http://127.0.0.1:7893"
  conda activate sam3
  python scripts/download_pretrained_weights.py
  python scripts/download_pretrained_weights.py --only efficientnet_b3
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WEIGHT_SPECS = {
    "resnet50": ("resnet50", "ResNet50_Weights", "IMAGENET1K_V1"),
    "resnet18": ("resnet18", "ResNet18_Weights", "IMAGENET1K_V1"),
    "efficientnet_b3": ("efficientnet_b3", "EfficientNet_B3_Weights", "IMAGENET1K_V1"),
    "convnext_t": ("convnext_tiny", "ConvNeXt_Tiny_Weights", "IMAGENET1K_V1"),
}

EFFICIENTNET_B3_URL = (
    "https://download.pytorch.org/models/efficientnet_b3_rwightman-b3899882.pth"
)
EFFICIENTNET_B3_FILE = "efficientnet_b3_rwightman-b3899882.pth"


def cache_dir() -> Path:
    torch_home = os.environ.get("TORCH_HOME", os.path.expanduser("~/.cache/torch"))
    return Path(torch_home) / "hub" / "checkpoints"


def download_via_torchvision(name: str) -> None:
    import torchvision.models as models

    fn, weights_cls, weights_ver = WEIGHT_SPECS[name]
    weights = getattr(models, weights_cls)
    weight_enum = getattr(weights, weights_ver)
    builder = getattr(models, fn)
    print(f"[torchvision] downloading {name} ({weight_enum.url}) ...")
    builder(weights=weight_enum)
    dest = cache_dir() / Path(weight_enum.url).name
    print(f"  -> cached at {dest}")


def download_efficientnet_curl() -> Path:
    """Fallback: wget/curl EfficientNet-B3 when torchvision download fails."""
    import urllib.request

    dest = cache_dir()
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / EFFICIENTNET_B3_FILE
    if out.exists() and out.stat().st_size > 1_000_000:
        print(f"[curl] already cached: {out}")
        return out

    print(f"[curl] fetching {EFFICIENTNET_B3_URL}")
    req = urllib.request.Request(
        EFFICIENTNET_B3_URL,
        headers={"User-Agent": "cvproject/1.0"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        out.write_bytes(resp.read())
    print(f"  -> saved {out} ({out.stat().st_size // 1024 // 1024} MB)")
    return out


def main():
    parser = argparse.ArgumentParser(description="Download pretrained backbone weights")
    parser.add_argument(
        "--only",
        nargs="*",
        choices=list(WEIGHT_SPECS.keys()),
        default=list(WEIGHT_SPECS.keys()),
    )
    args = parser.parse_args()

    print(f"TORCH cache: {cache_dir()}")
    print(f"http_proxy={os.environ.get('http_proxy', '(not set)')}")

    for name in args.only:
        try:
            download_via_torchvision(name)
        except Exception as exc:
            print(f"[warn] torchvision download failed for {name}: {exc}")
            if name == "efficientnet_b3":
                download_efficientnet_curl()

    b3 = cache_dir() / EFFICIENTNET_B3_FILE
    if b3.exists():
        print(f"\nEfficientNet-B3 ready: {b3}")
    else:
        print("\nEfficientNet-B3 not found. Set proxy and retry:")
        print('  export http_proxy="http://127.0.0.1:7893"')
        print('  export https_proxy="http://127.0.0.1:7893"')
        print("  python scripts/download_pretrained_weights.py --only efficientnet_b3")


if __name__ == "__main__":
    main()

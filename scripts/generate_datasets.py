"""
Generate two harder datasets from animal-90:

1. animal-90-multianimal  — copy-paste multiple animals onto natural backgrounds
2. animal-90-occlusion    — single animal with random translation + occluders

Each sample stores normalized xywh bbox annotations for localization training.
Per-class sample count is capped (default 30) to keep generation fast.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def list_class_images(root: Path) -> dict[str, list[Path]]:
    mapping = {}
    for cls_dir in sorted(root.iterdir()):
        if not cls_dir.is_dir():
            continue
        imgs = [p for p in cls_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
        if imgs:
            mapping[cls_dir.name] = imgs
    return mapping


def load_rgb(path: Path, size: Tuple[int, int] | None = None) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if size:
        img = img.resize(size, Image.Resampling.LANCZOS)
    return img


def random_background(w: int, h: int) -> Image.Image:
    """Procedural background: gradient + noise."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    c1 = np.random.randint(40, 180, size=3)
    c2 = np.random.randint(40, 180, size=3)
    for y in range(h):
        t = y / max(h - 1, 1)
        arr[y] = (c1 * (1 - t) + c2 * t).astype(np.uint8)
    noise = np.random.randint(-15, 15, arr.shape, dtype=np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def extract_foreground(img: Image.Image, threshold: int = 240) -> Image.Image:
    """Simple center-weighted crop as pseudo-foreground (no external segmentation)."""
    w, h = img.size
    side = int(min(w, h) * random.uniform(0.55, 0.85))
    cx, cy = w // 2, h // 2
    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    crop = img.crop((x1, y1, x1 + side, y1 + side))

    arr = np.array(crop)
    gray = arr.mean(axis=2)
    mask = gray < threshold
    arr[~mask] = [0, 0, 0]
    return Image.fromarray(arr)


def paste_animal(
    canvas: Image.Image,
    animal: Image.Image,
    max_scale: float = 0.35,
) -> Tuple[int, int, int, int]:
    cw, ch = canvas.size
    scale = random.uniform(0.12, max_scale)
    tw = int(cw * scale)
    th = int(animal.height * tw / max(animal.width, 1))
    animal = animal.resize((tw, th), Image.Resampling.LANCZOS)

    x = random.randint(0, max(cw - tw, 0))
    y = random.randint(0, max(ch - th, 0))
    canvas.paste(animal, (x, y))
    return x, y, tw, th


def generate_multianimal(
    source_root: Path,
    out_root: Path,
    samples_per_class: int = 30,
    canvas_size: Tuple[int, int] = (640, 480),
    animals_per_image: Tuple[int, int] = (2, 4),
) -> None:
    out_images = out_root / "images"
    out_ann = out_root / "annotations"
    out_images.mkdir(parents=True, exist_ok=True)
    out_ann.mkdir(parents=True, exist_ok=True)

    class_images = list_class_images(source_root)
    classes = list(class_images.keys())
    idx = 0

    for cls in classes:
        for _ in range(samples_per_class):
            canvas = random_background(*canvas_size)
            n_animals = random.randint(*animals_per_image)
            chosen = random.choices(classes, k=n_animals)
            objects = []

            for c in chosen:
                src = random.choice(class_images[c])
                fg = extract_foreground(load_rgb(src))
                x, y, w, h = paste_animal(canvas, fg)
                cw, ch = canvas.size
                objects.append({
                    "class": c,
                    "bbox": [x / cw, y / ch, w / cw, h / ch],
                })

            fname = f"multianimal_{idx:05d}.jpg"
            canvas.save(out_images / fname, quality=92)
            with open(out_ann / f"multianimal_{idx:05d}.json", "w", encoding="utf-8") as f:
                json.dump({
                    "file_name": fname,
                    "dataset": "animal-90-multianimal",
                    "objects": objects,
                }, f, indent=2)
            idx += 1

    print(f"Generated {idx} multi-animal images -> {out_root}")


def add_occluders(img: Image.Image, n: int = 3) -> Image.Image:
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for _ in range(n):
        ow = random.randint(w // 8, w // 4)
        oh = random.randint(h // 8, h // 4)
        ox = random.randint(0, max(w - ow, 0))
        oy = random.randint(0, max(h - oh, 0))
        color = tuple(random.randint(30, 220) for _ in range(3))
        draw.rectangle([ox, oy, ox + ow, oy + oh], fill=color)
    return img


def generate_occlusion(
    source_root: Path,
    out_root: Path,
    samples_per_class: int = 30,
    canvas_size: Tuple[int, int] = (640, 480),
) -> None:
    out_images = out_root / "images"
    out_ann = out_root / "annotations"
    out_images.mkdir(parents=True, exist_ok=True)
    out_ann.mkdir(parents=True, exist_ok=True)

    class_images = list_class_images(source_root)
    idx = 0

    for cls, imgs in class_images.items():
        for _ in range(min(samples_per_class, len(imgs) * 3)):
            src = random.choice(imgs)
            animal = load_rgb(src)
            fg = extract_foreground(animal)

            canvas = random_background(*canvas_size)
            x, y, w, h = paste_animal(canvas, fg, max_scale=0.45)

            if random.random() < 0.7:
                canvas = add_occluders(canvas, n=random.randint(1, 4))

            if random.random() < 0.5:
                canvas = canvas.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1.2)))

            enhancer = ImageEnhance.Brightness(canvas)
            canvas = enhancer.enhance(random.uniform(0.75, 1.25))

            cw, ch = canvas.size
            fname = f"occlusion_{idx:05d}.jpg"
            canvas.save(out_images / fname, quality=92)
            with open(out_ann / f"occlusion_{idx:05d}.json", "w", encoding="utf-8") as f:
                json.dump({
                    "file_name": fname,
                    "dataset": "animal-90-occlusion",
                    "objects": [{
                        "class": cls,
                        "bbox": [x / cw, y / ch, w / cw, h / ch],
                    }],
                }, f, indent=2)
            idx += 1

    print(f"Generated {idx} occlusion images -> {out_root}")


def main():
    parser = argparse.ArgumentParser(description="Generate derived animal datasets")
    parser.add_argument("--source", default="./data/animal-90", help="animal-90 root")
    parser.add_argument("--out-multianimal", default="./data/animal-90-multianimal")
    parser.add_argument("--out-occlusion", default="./data/animal-90-occlusion")
    parser.add_argument("--samples-per-class", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(
            f"Source dataset not found: {source}\n"
            "Place animal-90 under data/animal-90/ with one folder per class."
        )

    generate_multianimal(source, Path(args.out_multianimal), args.samples_per_class)
    generate_occlusion(source, Path(args.out_occlusion), args.samples_per_class)


if __name__ == "__main__":
    main()

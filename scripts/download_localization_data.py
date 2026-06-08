"""
Download a small multi-animal localization dataset for training.

1. Ensures animal-90-multianimal exists (generates from animal-90 if needed).
2. Downloads Snapshot Serengeti bbox metadata + a subset of multi-animal images
   (~300 images with >=2 annotated animals) from LILA Azure blob storage.

Usage:
  python scripts/download_localization_data.py
  python scripts/download_localization_data.py --max-images 500
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SERENGETI_BBOX_ZIP = (
    "https://storage.googleapis.com/public-datasets-lila/snapshotserengeti-v-2-0/"
    "SnapshotSerengetiBboxes_20190903.json.zip"
)
SERENGETI_IMAGE_BASE = (
    "https://storage.googleapis.com/public-datasets-lila/snapshotserengeti-unzipped/"
)


def ensure_multianimal(source: Path, out: Path, per_class: int = 30) -> None:
    if out.exists() and (out / "images").exists() and len(list((out / "images").glob("*"))) > 100:
        print(f"[multianimal] already present: {out} ({len(list((out / 'images').glob('*')))} images)")
        return

    if not source.exists():
        print(f"[multianimal] skip generation: source not found at {source}")
        return

    print(f"[multianimal] generating from {source} -> {out}")
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "generate_datasets.py"),
        "--source",
        str(source),
        "--out-multianimal",
        str(out),
        "--samples-per-class",
        str(per_class),
    ]
    subprocess.run(cmd, check=True)


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[download] cached {dest.name}")
        return
    print(f"[download] fetching {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "cvproject/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    dest.write_bytes(data)


def extract_bbox_json(zip_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        json_names = [n for n in zf.namelist() if n.lower().endswith(".json")]
        if not json_names:
            raise RuntimeError(f"No JSON in {zip_path}")
        target = out_dir / Path(json_names[0]).name
        if not target.exists():
            with zf.open(json_names[0]) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
        return target


def select_multianimal_images(bbox_json: Path, min_objects: int, max_images: int) -> tuple[dict, list[dict]]:
    with open(bbox_json, encoding="utf-8") as f:
        data = json.load(f)

    images = {img["id"]: img for img in data.get("images", [])}
    by_image: dict[int, list] = defaultdict(list)
    for ann in data.get("annotations", []):
        if "bbox" in ann:
            by_image[ann["image_id"]].append(ann)

    candidates = []
    for img_id, anns in by_image.items():
        if len(anns) < min_objects:
            continue
        img_info = images.get(img_id)
        if img_info is None:
            continue
        candidates.append((len(anns), img_id, img_info, anns))

    candidates.sort(key=lambda x: (-x[0], x[1]))
    selected = candidates[:max_images]

    subset_images = []
    subset_anns = []
    id_map = {}
    for new_id, (_, old_id, img_info, anns) in enumerate(selected):
        id_map[old_id] = new_id
        subset_images.append({**img_info, "id": new_id})
        for ann in anns:
            subset_anns.append({**ann, "image_id": new_id})

    subset = {
        "images": subset_images,
        "annotations": subset_anns,
        "categories": data.get("categories", []),
        "info": {"description": "Serengeti multi-animal subset for localization training"},
    }
    return subset, selected


def download_serengeti_subset(out_dir: Path, max_images: int, min_objects: int) -> None:
    cache_dir = out_dir / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    zip_path = cache_dir / "serengeti_bboxes.zip"
    download_file(SERENGETI_BBOX_ZIP, zip_path)
    bbox_json = extract_bbox_json(zip_path, out_dir)

    subset, selected = select_multianimal_images(bbox_json, min_objects, max_images)
    subset_json = out_dir / "serengeti_multianimal_subset.json"
    with open(subset_json, "w", encoding="utf-8") as f:
        json.dump(subset, f)

    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    for _, _, img_info, _ in selected:
        file_name = img_info["file_name"]
        rel = file_name.lstrip("/")
        dest = images_dir / Path(rel).name
        if dest.exists() and dest.stat().st_size > 0:
            downloaded += 1
            continue
        url = SERENGETI_IMAGE_BASE + rel
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "cvproject/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                dest.write_bytes(resp.read())
            downloaded += 1
            if downloaded % 50 == 0:
                print(f"[serengeti] downloaded {downloaded}/{len(selected)} images")
        except Exception as exc:
            print(f"[serengeti] failed {rel}: {exc}")

    print(f"[serengeti] subset ready: {subset_json}  images={downloaded}/{len(selected)}")


def main():
    parser = argparse.ArgumentParser(description="Download multi-animal localization datasets")
    parser.add_argument("--animal90", default=str(ROOT / "data" / "animal-90"))
    parser.add_argument("--multianimal-out", default=str(ROOT / "data" / "animal-90-multianimal"))
    parser.add_argument("--serengeti-out", default=str(ROOT / "data" / "serengeti-multianimal"))
    parser.add_argument("--per-class", type=int, default=30, dest="samples_per_class")
    parser.add_argument("--max-images", type=int, default=300)
    parser.add_argument("--min-objects", type=int, default=2)
    parser.add_argument("--skip-serengeti", action="store_true")
    args = parser.parse_args()

    ensure_multianimal(Path(args.animal90), Path(args.multianimal_out), args.samples_per_class)

    if not args.skip_serengeti:
        download_serengeti_subset(
            Path(args.serengeti_out),
            max_images=args.max_images,
            min_objects=args.min_objects,
        )

    print("\nLocalization data ready. Train with:")
    print("  python scripts/train.py --data-path ./data/animal-90 \\")
    print("    --loc-data-paths ./data/animal-90-multianimal ./data/serengeti-multianimal")


if __name__ == "__main__":
    main()

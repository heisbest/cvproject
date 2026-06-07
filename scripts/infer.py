"""CLI inference using saved weights."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.pipeline import AnimalRecognitionPipeline
from utils.paths import weights_dir_for_dataset


def main():
    parser = argparse.ArgumentParser(description="Run inference on a single image")
    parser.add_argument("--image", required=True)
    parser.add_argument("--weights-dir", default=None)
    parser.add_argument("--data-path", default=None, help="Infer weights dir from training data path")
    parser.add_argument("--output", default=None, help="Save visualization")
    parser.add_argument("--no-locateanything", action="store_true")
    args = parser.parse_args()

    if args.weights_dir:
        weights_dir = Path(args.weights_dir)
    elif args.data_path:
        weights_dir = weights_dir_for_dataset(args.data_path)
    else:
        parser.error("Provide --weights-dir or --data-path")

    pipe = AnimalRecognitionPipeline(
        weights_dir,
        use_locateanything=not args.no_locateanything,
    )
    results = pipe.predict(args.image)

    print(f"\nDetections for {args.image}:")
    for i, r in enumerate(results):
        xywh = r["bbox_xywh"]
        print(f"  [{i}] xywh=({xywh[0]:.3f},{xywh[1]:.3f},{xywh[2]:.3f},{xywh[3]:.3f}) "
              f"class={r['class']} prob={r['confidence']:.4f}")

    if args.output:
        from PIL import Image

        img = Image.open(args.image).convert("RGB")
        vis = pipe.visualize(img, results)
        vis.save(args.output)
        print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()

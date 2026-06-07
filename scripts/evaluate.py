"""Evaluate localization and classification on a user-specified eval set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.evaluation import run_evaluation
from utils.paths import weights_dir_for_dataset


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained weights on an eval set")
    parser.add_argument("--weights-dir", default=None, help="Weights directory")
    parser.add_argument("--data-path", default=None, help="Training data path (to infer weights dir)")
    parser.add_argument("--eval-path", required=True, help="Evaluation dataset root")
    parser.add_argument("--dataset-type", choices=["animal90", "serengeti", "auto"], default="auto")
    args = parser.parse_args()

    if args.weights_dir:
        weights_dir = Path(args.weights_dir)
    elif args.data_path:
        weights_dir = weights_dir_for_dataset(args.data_path)
    else:
        parser.error("Provide --weights-dir or --data-path")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Weights: {weights_dir}")
    print(f"Eval set: {args.eval_path}")

    run_evaluation(
        weights_dir=weights_dir,
        eval_path=args.eval_path,
        dataset_type=args.dataset_type,
        device=device,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prepare the normalized PNG dataset once for HW3 experiments.

This is a convenience wrapper around convert_hw3_to_coco.py. It creates:

  data_norm_png/
  annotations_norm_png/

Training configs that use normalized PNGs read those folders directly, so the
TIFF -> p1-p99 normalize -> PNG conversion is paid once, not every epoch.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


REQUIRED_RELATIVE_PATHS = (
    "annotations_norm_png/instances_hw3_train.json",
    "annotations_norm_png/instances_hw3_val.json",
    "annotations_norm_png/image_info_hw3_test.json",
    "annotations_norm_png/split_hw3.json",
    "data_norm_png/train",
    "data_norm_png/val",
    "data_norm_png/test_release",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "annotations_norm_png")
    parser.add_argument("--export-images-dir", type=Path, default=REPO_ROOT / "data_norm_png")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-area", type=int, default=1)
    parser.add_argument("--simplify-eps", type=float, default=0.0)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if all normalized PNG dataset outputs already exist.",
    )
    return parser.parse_args()


def is_prepared(args: argparse.Namespace) -> bool:
    custom_outputs = (
        args.out_dir != REPO_ROOT / "annotations_norm_png"
        or args.export_images_dir != REPO_ROOT / "data_norm_png"
    )
    if custom_outputs:
        required = [
            args.out_dir / "instances_hw3_train.json",
            args.out_dir / "instances_hw3_val.json",
            args.out_dir / "image_info_hw3_test.json",
            args.out_dir / "split_hw3.json",
            args.export_images_dir / "train",
            args.export_images_dir / "val",
            args.export_images_dir / "test_release",
        ]
    else:
        required = [REPO_ROOT / path for path in REQUIRED_RELATIVE_PATHS]
    return all(path.exists() for path in required)


def main() -> None:
    args = parse_args()
    if is_prepared(args) and not args.force:
        print("Normalized PNG dataset already exists. Skipping conversion.")
        print("Use --force to rebuild it.")
        return

    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "convert_hw3_to_coco.py"),
        "--data-root",
        str(args.data_root),
        "--val-ratio",
        str(args.val_ratio),
        "--seed",
        str(args.seed),
        "--min-area",
        str(args.min_area),
        "--simplify-eps",
        str(args.simplify_eps),
        "--out-dir",
        str(args.out_dir),
        "--export-images-dir",
        str(args.export_images_dir),
        "--percentile-normalize",
    ]
    subprocess.check_call(cmd, cwd=REPO_ROOT)
    print("\nPrepared normalized PNG dataset:")
    print(f"  annotations: {args.out_dir}")
    print(f"  images:      {args.export_images_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prepare normalized PNG images with COCO RLE mask annotations for Exp4.

This mirrors the stronger classmate-style data conversion:

  raw TIFF -> p1-p99 normalized 3-channel PNG
  instance masks -> compressed COCO RLE instead of polygon contours

Training configs that use this dataset read:

  data_norm_png_rle/
  annotations_norm_png_rle/
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "annotations_norm_png_rle"
    )
    parser.add_argument(
        "--export-images-dir", type=Path, default=REPO_ROOT / "data_norm_png_rle"
    )
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-area",
        type=int,
        default=6,
        help="Ignore extremely tiny instances, matching the classmate pipeline.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def is_prepared(args: argparse.Namespace) -> bool:
    required = [
        args.out_dir / "instances_hw3_train.json",
        args.out_dir / "instances_hw3_val.json",
        args.out_dir / "image_info_hw3_test.json",
        args.out_dir / "split_hw3.json",
        args.export_images_dir / "train",
        args.export_images_dir / "val",
        args.export_images_dir / "test_release",
    ]
    return all(path.exists() for path in required)


def main() -> None:
    args = parse_args()
    if is_prepared(args) and not args.force:
        print("Normalized PNG + RLE dataset already exists. Skipping conversion.")
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
        "--out-dir",
        str(args.out_dir),
        "--export-images-dir",
        str(args.export_images_dir),
        "--percentile-normalize",
        "--segmentation-format",
        "rle",
    ]
    subprocess.check_call(cmd, cwd=REPO_ROOT)
    print("\nPrepared normalized PNG + RLE dataset:")
    print(f"  annotations: {args.out_dir}")
    print(f"  images:      {args.export_images_dir}")


if __name__ == "__main__":
    main()

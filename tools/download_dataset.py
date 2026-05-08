#!/usr/bin/env python3
"""Download and extract the HW3 dataset from Google Drive."""

from __future__ import annotations

import argparse
import shutil
import tarfile
import zipfile
from pathlib import Path

import gdown


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URL = "https://drive.google.com/file/d/1uCnJ3LrsBHOeQoJDoe4Yg8H32VuQJodv/view"


def extract_archive(archive_path: Path, out_dir: Path) -> None:
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(out_dir)
        return
    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as archive:
            archive.extractall(out_dir)
        return
    raise ValueError(f"Unsupported archive format: {archive_path}")


def normalize_data_layout(out_dir: Path) -> None:
    """Move nested data contents to out_dir when archives contain a top folder."""
    if (out_dir / "train").exists() and (out_dir / "test_release").exists():
        return
    candidates = [
        path
        for path in out_dir.iterdir()
        if path.is_dir() and (path / "train").exists() and (path / "test_release").exists()
    ]
    if not candidates:
        return
    source = candidates[0]
    for child in source.iterdir():
        target = out_dir / child.name
        if target.exists():
            continue
        shutil.move(str(child), str(target))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--archive", type=Path, default=REPO_ROOT / "hw3_dataset")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    archive_path = gdown.download(args.url, str(args.archive), quiet=False, fuzzy=True)
    if archive_path is None:
        raise RuntimeError("gdown failed to download the dataset.")
    archive = Path(archive_path)
    extract_archive(archive, args.out_dir)
    normalize_data_layout(args.out_dir)
    print(f"Dataset extracted to {args.out_dir}")


if __name__ == "__main__":
    main()

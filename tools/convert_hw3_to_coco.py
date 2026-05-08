#!/usr/bin/env python3
"""Convert HW3 TIFF masks into COCO instance segmentation annotations."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

import cv2
import numpy as np
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = [
    {"id": 1, "name": "class1", "supercategory": "cell"},
    {"id": 2, "name": "class2", "supercategory": "cell"},
    {"id": 3, "name": "class3", "supercategory": "cell"},
    {"id": 4, "name": "class4", "supercategory": "cell"},
]


def read_tif(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to read {path}")
    return image


def polygons_from_mask(binary_mask: np.ndarray, simplify_eps: float) -> List[List[float]]:
    mask = binary_mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: List[List[float]] = []
    for contour in contours:
        if simplify_eps > 0:
            contour = cv2.approxPolyDP(contour, epsilon=simplify_eps, closed=True)
        if contour.shape[0] < 3:
            continue
        polygon = contour.reshape(-1, 2).astype(float).flatten().tolist()
        if len(polygon) >= 6:
            polygons.append(polygon)
    return polygons


def bbox_from_mask(binary_mask: np.ndarray) -> List[int]:
    ys, xs = np.where(binary_mask)
    x_min = int(xs.min())
    y_min = int(ys.min())
    width = int(xs.max() - x_min + 1)
    height = int(ys.max() - y_min + 1)
    return [x_min, y_min, width, height]


def image_record(image_id: int, image_dir: Path) -> Dict:
    image = read_tif(image_dir / "image.tif")
    height, width = image.shape[:2]
    return {
        "id": image_id,
        "file_name": f"{image_dir.name}/image.tif",
        "height": int(height),
        "width": int(width),
    }


def iter_annotations(
    image_id: int, image_dir: Path, simplify_eps: float, min_area: int
) -> Iterable[Dict]:
    for category_id in range(1, 5):
        mask_path = image_dir / f"class{category_id}.tif"
        if not mask_path.exists():
            continue
        mask = read_tif(mask_path)
        instance_ids = np.unique(mask)
        instance_ids = instance_ids[instance_ids != 0]
        for instance_id in instance_ids:
            binary_mask = mask == instance_id
            area = int(binary_mask.sum())
            if area < min_area:
                continue
            segmentation = polygons_from_mask(binary_mask, simplify_eps)
            if not segmentation:
                continue
            yield {
                "image_id": image_id,
                "category_id": category_id,
                "bbox": bbox_from_mask(binary_mask),
                "area": area,
                "segmentation": segmentation,
                "iscrowd": 0,
            }


def make_coco(image_dirs: Sequence[Path], simplify_eps: float, min_area: int) -> Dict:
    images: List[Dict] = []
    annotations: List[Dict] = []
    ann_id = 1
    for image_id, image_dir in enumerate(tqdm(image_dirs, desc="Converting"), start=1):
        images.append(image_record(image_id, image_dir))
        for annotation in iter_annotations(image_id, image_dir, simplify_eps, min_area):
            annotation["id"] = ann_id
            annotations.append(annotation)
            ann_id += 1
    return {
        "images": images,
        "annotations": annotations,
        "categories": CATEGORIES,
        "licenses": [],
        "info": {"description": "VRDL HW3 in COCO instance format"},
    }


def make_test_info(mapping_path: Path) -> Dict:
    images = json.loads(mapping_path.read_text())
    return {
        "images": images,
        "annotations": [],
        "categories": CATEGORIES,
        "licenses": [],
        "info": {"description": "VRDL HW3 test image info"},
    }


def split_train_val(
    image_dirs: Sequence[Path], val_ratio: float, seed: int
) -> Tuple[List[Path], List[Path]]:
    image_dirs = list(image_dirs)
    random.Random(seed).shuffle(image_dirs)
    val_count = max(1, round(len(image_dirs) * val_ratio))
    val_dirs = sorted(image_dirs[:val_count])
    train_dirs = sorted(image_dirs[val_count:])
    return train_dirs, val_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "annotations",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-area", type=int, default=1)
    parser.add_argument(
        "--simplify-eps",
        type=float,
        default=0.0,
        help="Polygon simplification epsilon. Keep 0.0 for maximum mask detail.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_root = args.data_root / "train"
    image_dirs = sorted(path for path in train_root.iterdir() if path.is_dir())
    train_dirs, val_dirs = split_train_val(image_dirs, args.val_ratio, args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_json = make_coco(train_dirs, args.simplify_eps, args.min_area)
    val_json = make_coco(val_dirs, args.simplify_eps, args.min_area)
    (args.out_dir / "instances_hw3_train.json").write_text(json.dumps(train_json))
    (args.out_dir / "instances_hw3_val.json").write_text(json.dumps(val_json))
    (args.out_dir / "image_info_hw3_test.json").write_text(
        json.dumps(make_test_info(args.data_root / "test_image_name_to_ids.json"))
    )
    (args.out_dir / "split_hw3.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "val_ratio": args.val_ratio,
                "train": [path.name for path in train_dirs],
                "val": [path.name for path in val_dirs],
            },
            indent=2,
        )
    )
    print(
        "Wrote "
        f"{len(train_json['images'])} train images / {len(train_json['annotations'])} annotations, "
        f"{len(val_json['images'])} val images / {len(val_json['annotations'])} annotations."
    )


if __name__ == "__main__":
    main()

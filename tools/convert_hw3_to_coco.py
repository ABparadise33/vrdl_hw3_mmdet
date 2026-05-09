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


def to_3ch(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] in (1, 2, 3, 4) and image.shape[-1] not in (1, 2, 3, 4):
        image = np.transpose(image, (1, 2, 0))

    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    elif image.ndim == 3 and image.shape[-1] == 1:
        image = np.concatenate([image, image, image], axis=-1)
    elif image.ndim == 3 and image.shape[-1] == 2:
        image = np.concatenate([image, image[:, :, :1]], axis=-1)
    elif image.ndim == 3 and image.shape[-1] >= 4:
        image = image[:, :, :3]
    return image


def normalize_percentile(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    p1, p99 = np.percentile(image, (1, 99))
    image = np.clip(image, p1, p99)
    span = p99 - p1
    if span > 1e-6:
        image = (image - p1) / span
    else:
        image = image / (image.max() + 1e-6)
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def prepare_export_image(image: np.ndarray, normalize: bool) -> np.ndarray:
    image = to_3ch(image)
    if normalize:
        image = normalize_percentile(image)
    elif image.dtype != np.uint8:
        info = np.iinfo(image.dtype) if np.issubdtype(image.dtype, np.integer) else None
        if info is not None and info.max > 255:
            image = (image.astype(np.float32) / info.max * 255.0).clip(0, 255).astype(np.uint8)
        else:
            image = image.clip(0, 255).astype(np.uint8)
    return image


def export_image(
    image_dir: Path,
    split: str,
    export_dir: Path,
    normalize: bool,
) -> str:
    image = prepare_export_image(read_tif(image_dir / "image.tif"), normalize)
    file_name = f"{image_dir.name}.png"
    out_path = export_dir / split / file_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), image):
        raise RuntimeError(f"Failed to write {out_path}")
    return file_name


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


def image_record(
    image_id: int,
    image_dir: Path,
    split: str,
    export_dir: Path | None,
    normalize: bool,
) -> Dict:
    image = read_tif(image_dir / "image.tif")
    height, width = image.shape[:2]
    if export_dir is not None:
        file_name = export_image(image_dir, split, export_dir, normalize)
    else:
        file_name = f"{image_dir.name}/image.tif"
    return {
        "id": image_id,
        "file_name": file_name,
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


def make_coco(
    image_dirs: Sequence[Path],
    split: str,
    simplify_eps: float,
    min_area: int,
    export_dir: Path | None,
    normalize: bool,
) -> Dict:
    images: List[Dict] = []
    annotations: List[Dict] = []
    ann_id = 1
    for image_id, image_dir in enumerate(tqdm(image_dirs, desc="Converting"), start=1):
        images.append(image_record(image_id, image_dir, split, export_dir, normalize))
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


def make_test_info(
    mapping_path: Path,
    test_root: Path,
    export_dir: Path | None,
    normalize: bool,
) -> Dict:
    images = json.loads(mapping_path.read_text())
    if export_dir is not None:
        converted_images = []
        out_dir = export_dir / "test_release"
        out_dir.mkdir(parents=True, exist_ok=True)
        for info in tqdm(images, desc="Converting test"):
            src_path = test_root / info["file_name"]
            image = prepare_export_image(read_tif(src_path), normalize)
            file_name = Path(info["file_name"]).with_suffix(".png").name
            out_path = out_dir / file_name
            if not cv2.imwrite(str(out_path), image):
                raise RuntimeError(f"Failed to write {out_path}")
            converted = dict(info)
            converted["file_name"] = file_name
            converted_images.append(converted)
        images = converted_images
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
    parser.add_argument(
        "--export-images-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory for converted PNG images. If set, train/val/test "
            "COCO file_name fields point to this exported layout instead of raw TIFFs."
        ),
    )
    parser.add_argument(
        "--percentile-normalize",
        action="store_true",
        help="Clip image intensities to p1-p99 and rescale to uint8 before PNG export.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_root = args.data_root / "train"
    test_root = args.data_root / "test_release"
    image_dirs = sorted(path for path in train_root.iterdir() if path.is_dir())
    train_dirs, val_dirs = split_train_val(image_dirs, args.val_ratio, args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_json = make_coco(
        train_dirs,
        "train",
        args.simplify_eps,
        args.min_area,
        args.export_images_dir,
        args.percentile_normalize,
    )
    val_json = make_coco(
        val_dirs,
        "val",
        args.simplify_eps,
        args.min_area,
        args.export_images_dir,
        args.percentile_normalize,
    )
    (args.out_dir / "instances_hw3_train.json").write_text(json.dumps(train_json))
    (args.out_dir / "instances_hw3_val.json").write_text(json.dumps(val_json))
    (args.out_dir / "image_info_hw3_test.json").write_text(
        json.dumps(
            make_test_info(
                args.data_root / "test_image_name_to_ids.json",
                test_root,
                args.export_images_dir,
                args.percentile_normalize,
            )
        )
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

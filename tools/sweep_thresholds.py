#!/usr/bin/env python3
"""Sweep validation score thresholds and report COCO AP changes."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from tqdm import tqdm

from infer_submit import (
    encode_mask,
    instances_from_result,
    load_detector,
    mask_to_xywh,
    run_inference,
    run_manual_tta,
    xyxy_to_xywh,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_THRESHOLDS = [
    0.0,
    0.001,
    0.002,
    0.003,
    0.005,
    0.0075,
    0.01,
    0.0125,
    0.015,
    0.0175,
    0.02,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ann-file", type=Path, default=REPO_ROOT / "annotations/instances_hw3_val.json")
    parser.add_argument("--img-root", type=Path, default=REPO_ROOT / "data/train")
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=DEFAULT_THRESHOLDS,
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--exp-name", default=None)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--tta-scales", nargs="+", type=int, default=[800, 1000, 1200])
    parser.add_argument("--tta-no-flip", action="store_true")
    parser.add_argument("--tta-nms-iou", type=float, default=0.5)
    parser.add_argument("--max-per-img", type=int, default=300)
    parser.add_argument("--bbox-from-mask", action="store_true")
    return parser.parse_args()


def load_val_images(ann_file: Path) -> List[Dict]:
    data = json.loads(ann_file.read_text())
    return data["images"]


def collect_predictions(
    model,
    images: List[Dict],
    img_root: Path,
    use_tta: bool,
    tta_scales: Sequence[int],
    tta_flip: bool,
    tta_nms_iou: float,
    max_per_img: int,
    bbox_from_mask: bool,
) -> List[Dict]:
    predictions = []
    counts_by_image: List[int] = []
    for image_info in tqdm(images, desc="Infer val"):
        image_path = img_root / image_info["file_name"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read {image_path}")
        if use_tta:
            boxes, scores, labels, masks = run_manual_tta(
                model,
                image,
                scales=tta_scales,
                use_flip=tta_flip,
                nms_iou=tta_nms_iou,
                max_per_img=max_per_img,
            )
        else:
            result = run_inference(model, image_path)
            boxes, scores, labels, masks = instances_from_result(result)
        if masks is None:
            counts_by_image.append(0)
            continue
        image_count = 0
        for box, score, label, mask in zip(boxes, scores, labels, masks):
            binary_mask = mask.astype(bool)
            bbox = mask_to_xywh(binary_mask) if bbox_from_mask else xyxy_to_xywh(box)
            if bbox is None:
                continue
            predictions.append(
                {
                    "image_id": int(image_info["id"]),
                    "bbox": bbox,
                    "score": float(score),
                    "category_id": int(label) + 1,
                    "segmentation": encode_mask(binary_mask),
                }
            )
            image_count += 1
        counts_by_image.append(image_count)
    if counts_by_image:
        print(
            "Raw predictions/image: "
            f"mean={np.mean(counts_by_image):.1f}, "
            f"median={np.median(counts_by_image):.1f}, "
            f"max={np.max(counts_by_image)}"
        )
    return predictions


def evaluate_predictions(ann_file: Path, predictions: List[Dict]) -> Dict[str, float]:
    from contextlib import redirect_stdout
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(str(ann_file))
    if predictions:
        coco_dt = coco_gt.loadRes(predictions)
    else:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump([], f)
            empty_path = f.name
        coco_dt = coco_gt.loadRes(empty_path)

    output: Dict[str, float] = {}
    category_names = {
        category["id"]: category["name"] for category in coco_gt.loadCats(coco_gt.getCatIds())
    }
    for metric in ("bbox", "segm"):
        evaluator = COCOeval(coco_gt, coco_dt, metric)
        evaluator.params.maxDets = [100, 300, 1000]
        with open(os.devnull, "w") as devnull, redirect_stdout(devnull):
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
        output[f"{metric}_mAP"] = float(evaluator.stats[0])
        output[f"{metric}_mAP_50"] = float(evaluator.stats[1])
        output[f"{metric}_mAP_75"] = float(evaluator.stats[2])
        output[f"{metric}_mAP_s"] = float(evaluator.stats[3])
        output[f"{metric}_mAP_m"] = float(evaluator.stats[4])
        output[f"{metric}_mAP_l"] = float(evaluator.stats[5])

        precisions = evaluator.eval["precision"]
        for cat_index, cat_id in enumerate(evaluator.params.catIds):
            name = category_names.get(cat_id, f"class{cat_id}")
            all_area = 0
            max_det = len(evaluator.params.maxDets) - 1
            class_precision = precisions[:, :, cat_index, all_area, max_det]
            valid = class_precision[class_precision > -1]
            output[f"{metric}_{name}_AP"] = float(valid.mean()) if valid.size else float("nan")
            class_precision_50 = precisions[0, :, cat_index, all_area, max_det]
            valid_50 = class_precision_50[class_precision_50 > -1]
            output[f"{metric}_{name}_AP50"] = (
                float(valid_50.mean()) if valid_50.size else float("nan")
            )
    return output


def main() -> None:
    args = parse_args()
    out_path = args.out
    if out_path is None:
        exp_name = args.exp_name or Path(args.checkpoint).parent.name
        out_path = REPO_ROOT / "results" / exp_name / "threshold_sweep.csv"

    model = load_detector(args.config, args.checkpoint, args.device)
    images = load_val_images(args.ann_file)
    predictions = collect_predictions(
        model,
        images,
        args.img_root,
        use_tta=args.tta,
        tta_scales=args.tta_scales,
        tta_flip=not args.tta_no_flip,
        tta_nms_iou=args.tta_nms_iou,
        max_per_img=args.max_per_img,
        bbox_from_mask=args.bbox_from_mask,
    )
    print(f"Collected {len(predictions)} raw predictions.")

    rows = []
    for threshold in args.thresholds:
        filtered = [pred for pred in predictions if pred["score"] >= threshold]
        class_counts = {
            f"num_class{class_id}": sum(
                pred["category_id"] == class_id for pred in filtered
            )
            for class_id in range(1, 5)
        }
        metrics = evaluate_predictions(args.ann_file, filtered)
        row = {
            "score_thr": threshold,
            "num_predictions": len(filtered),
            **class_counts,
            **metrics,
        }
        rows.append(row)
        print(
            f"thr={threshold:.3f} n={len(filtered)} "
            f"segm50={metrics['segm_mAP_50']:.4f} "
            f"segmAP={metrics['segm_mAP']:.4f} "
            f"bbox50={metrics['bbox_mAP_50']:.4f}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote threshold sweep to {out_path}")


if __name__ == "__main__":
    main()

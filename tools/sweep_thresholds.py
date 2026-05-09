#!/usr/bin/env python3
"""Sweep validation score thresholds and report COCO AP changes."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from tqdm import tqdm

from infer_submit import encode_mask, instances_from_result, load_detector, run_inference, xyxy_to_xywh


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_THRESHOLDS = [0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]


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
    return parser.parse_args()


def load_val_images(ann_file: Path) -> List[Dict]:
    data = json.loads(ann_file.read_text())
    return data["images"]


def collect_predictions(model, images: List[Dict], img_root: Path) -> List[Dict]:
    predictions = []
    for image_info in tqdm(images, desc="Infer val"):
        image_path = img_root / image_info["file_name"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read {image_path}")
        result = run_inference(model, image_path)
        boxes, scores, labels, masks = instances_from_result(result)
        if masks is None:
            continue
        for box, score, label, mask in zip(boxes, scores, labels, masks):
            predictions.append(
                {
                    "image_id": int(image_info["id"]),
                    "bbox": xyxy_to_xywh(box),
                    "score": float(score),
                    "category_id": int(label) + 1,
                    "segmentation": encode_mask(mask.astype(bool)),
                }
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
    return output


def main() -> None:
    args = parse_args()
    out_path = args.out
    if out_path is None:
        exp_name = args.exp_name or Path(args.checkpoint).parent.name
        out_path = REPO_ROOT / "results" / exp_name / "threshold_sweep.csv"

    model = load_detector(args.config, args.checkpoint, args.device)
    images = load_val_images(args.ann_file)
    predictions = collect_predictions(model, images, args.img_root)
    print(f"Collected {len(predictions)} raw predictions.")

    rows = []
    for threshold in args.thresholds:
        filtered = [pred for pred in predictions if pred["score"] >= threshold]
        metrics = evaluate_predictions(args.ann_file, filtered)
        row = {"score_thr": threshold, "num_predictions": len(filtered), **metrics}
        rows.append(row)
        print(
            f"thr={threshold:.3f} n={len(filtered)} "
            f"segm50={metrics['segm_mAP_50']:.4f} bbox50={metrics['bbox_mAP_50']:.4f}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote threshold sweep to {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot a mask-IoU confusion matrix from COCO GT and prediction JSON."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils


CLASS_NAMES = ["class1", "class2", "class3", "class4", "background"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ann_file", type=Path)
    parser.add_argument("pred_json", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--iou-thr", type=float, default=0.5)
    parser.add_argument("--score-thr", type=float, default=0.05)
    return parser.parse_args()


def pred_to_rle(pred: Dict) -> Dict:
    rle = dict(pred["segmentation"])
    if isinstance(rle.get("counts"), str):
        rle["counts"] = rle["counts"].encode("utf-8")
    return rle


def mask_iou(rle_a: Dict, rle_b: Dict) -> float:
    return float(mask_utils.iou([rle_a], [rle_b], [0])[0, 0])


def load_predictions(path: Path, score_thr: float) -> Dict[int, List[Dict]]:
    predictions = json.loads(path.read_text())
    by_image: Dict[int, List[Dict]] = {}
    for pred in predictions:
        if float(pred.get("score", 0.0)) < score_thr:
            continue
        by_image.setdefault(int(pred["image_id"]), []).append(pred)
    for preds in by_image.values():
        preds.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return by_image


def compute_confusion(coco: COCO, predictions: Dict[int, List[Dict]], iou_thr: float) -> np.ndarray:
    matrix = np.zeros((5, 5), dtype=int)
    bg = 4

    for image_id in coco.getImgIds():
        ann_ids = coco.getAnnIds(imgIds=[image_id])
        anns = coco.loadAnns(ann_ids)
        gt_items = [
            {
                "category_id": int(ann["category_id"]),
                "rle": coco.annToRLE(ann),
                "matched": False,
            }
            for ann in anns
        ]

        for pred in predictions.get(image_id, []):
            pred_rle = pred_to_rle(pred)
            pred_cls = int(pred["category_id"]) - 1
            best_idx = None
            best_iou = 0.0
            for idx, gt in enumerate(gt_items):
                if gt["matched"]:
                    continue
                iou = mask_iou(pred_rle, gt["rle"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_idx is not None and best_iou >= iou_thr:
                gt_items[best_idx]["matched"] = True
                gt_cls = gt_items[best_idx]["category_id"] - 1
                matrix[gt_cls, pred_cls] += 1
            else:
                matrix[bg, pred_cls] += 1

        for gt in gt_items:
            if not gt["matched"]:
                matrix[int(gt["category_id"]) - 1, bg] += 1

    return matrix


def save_csv(matrix: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gt\\pred", *CLASS_NAMES])
        for name, row in zip(CLASS_NAMES, matrix):
            writer.writerow([name, *row.tolist()])


def save_plot(matrix: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(8, 7))
    plt.imshow(matrix, cmap="Blues")
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right")
    plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("Ground truth")
    plt.title("Mask IoU Confusion Matrix")
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            plt.text(x, y, str(matrix[y, x]), ha="center", va="center", fontsize=8)
    plt.colorbar(fraction=0.046, pad=0.04)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    coco = COCO(str(args.ann_file))
    predictions = load_predictions(args.pred_json, args.score_thr)
    matrix = compute_confusion(coco, predictions, args.iou_thr)
    save_csv(matrix, args.out_dir / "confusion_matrix.csv")
    save_plot(matrix, args.out_dir / "confusion_matrix.png")
    print(f"Wrote confusion matrix to {args.out_dir}")


if __name__ == "__main__":
    main()

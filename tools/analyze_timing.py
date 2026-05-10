#!/usr/bin/env python3
"""Summarize train/validation timing from an HW3 epoch_summary.csv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("epoch_summary", type=Path)
    parser.add_argument("--train-iters", type=int, default=23)
    parser.add_argument("--val-iters", type=int, default=31)
    return parser.parse_args()


def f(row: dict, key: str) -> float:
    try:
        return float(row.get(key, 0.0) or 0.0)
    except ValueError:
        return 0.0


def main() -> None:
    args = parse_args()
    with args.epoch_summary.open(newline="") as fp:
        rows = list(csv.DictReader(fp))
    if not rows:
        raise SystemExit(f"No rows in {args.epoch_summary}")

    train_data = [f(row, "train/data_time") for row in rows]
    train_total = [f(row, "train/time") for row in rows]
    train_model = [max(0.0, total - data) for data, total in zip(train_data, train_total)]

    val_data = [f(row, "val/data_time") for row in rows]
    val_total = [f(row, "val/time") for row in rows]
    val_model = [max(0.0, total - data) for data, total in zip(val_data, val_total)]

    train_data_hours = sum(train_data) * args.train_iters / 3600
    train_model_hours = sum(train_model) * args.train_iters / 3600
    val_data_hours = sum(val_data) * args.val_iters / 3600
    val_model_hours = sum(val_model) * args.val_iters / 3600
    total_hours = train_data_hours + train_model_hours + val_data_hours + val_model_hours

    best50 = max(rows, key=lambda row: f(row, "val/coco/segm_mAP_50"))
    bestmap = max(rows, key=lambda row: f(row, "val/coco/segm_mAP"))
    lowloss = min(rows, key=lambda row: f(row, "train/loss"))

    print(f"File: {args.epoch_summary}")
    print(f"Epochs: {len(rows)}")
    print(f"Best segm_mAP_50: epoch {best50['epoch']} = {f(best50, 'val/coco/segm_mAP_50'):.4f}")
    print(f"Best segm_mAP:    epoch {bestmap['epoch']} = {f(bestmap, 'val/coco/segm_mAP'):.4f}")
    print(f"Lowest loss:      epoch {lowloss['epoch']} = {f(lowloss, 'train/loss'):.4f}")
    print()
    print("Average train iteration:")
    print(f"  data_time:  {mean(train_data):.2f}s")
    print(f"  model_time: {mean(train_model):.2f}s")
    print(f"  total:      {mean(train_total):.2f}s")
    print()
    print("Average validation iteration:")
    print(f"  data_time:  {mean(val_data):.2f}s")
    print(f"  model/eval: {mean(val_model):.2f}s")
    print(f"  total:      {mean(val_total):.2f}s")
    print()
    print("Estimated total time:")
    print(f"  train data:       {train_data_hours:.2f}h")
    print(f"  train model/GPU:  {train_model_hours:.2f}h")
    print(f"  val data:         {val_data_hours:.2f}h")
    print(f"  val model/eval:   {val_model_hours:.2f}h")
    print(f"  total:            {total_hours:.2f}h")


if __name__ == "__main__":
    main()

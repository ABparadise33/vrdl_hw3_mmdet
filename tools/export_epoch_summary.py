#!/usr/bin/env python3
"""Export per-epoch train losses and validation AP metrics from a work_dir."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work_dir", type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path. Default: <work_dir>/epoch_summary.csv.",
    )
    return parser.parse_args()


def load_json_records(path: Path) -> List[Dict]:
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def find_records(work_dir: Path) -> List[Dict]:
    records = []
    for path in sorted(work_dir.glob("**/vis_data/scalars.json")):
        records.extend(load_json_records(path))
    if not records:
        raise SystemExit(f"No scalars.json found under {work_dir}")
    return records


def is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def average_rows(rows: List[Dict], keys: List[str]) -> Dict[str, float]:
    averaged = {}
    for key in keys:
        values = [float(row[key]) for row in rows if is_number(row.get(key))]
        if values:
            averaged[key] = sum(values) / len(values)
    return averaged


def main() -> None:
    args = parse_args()
    out_path = args.out or REPO_ROOT / "results" / args.work_dir.name / "epoch_summary.csv"
    records = find_records(args.work_dir)

    train_by_epoch: Dict[int, List[Dict]] = defaultdict(list)
    val_by_epoch: Dict[int, Dict] = {}

    for record in records:
        if "loss" in record and "epoch" in record:
            train_by_epoch[int(record["epoch"])].append(record)
        if "coco/segm_mAP_50" in record and "step" in record:
            val_by_epoch[int(record["step"])] = record

    train_keys = sorted(
        {
            key
            for rows in train_by_epoch.values()
            for row in rows
            for key, value in row.items()
            if is_number(value) and key not in {"epoch", "iter", "step"}
        }
    )
    val_keys = sorted(
        {
            key
            for row in val_by_epoch.values()
            for key, value in row.items()
            if is_number(value) and key not in {"step"}
        }
    )

    epochs = sorted(set(train_by_epoch.keys()) | set(val_by_epoch.keys()))
    rows = []
    for epoch in epochs:
        row: Dict[str, float | int | str] = {"epoch": epoch}
        train_avg = average_rows(train_by_epoch.get(epoch, []), train_keys)
        for key, value in train_avg.items():
            row[f"train/{key}"] = value
        val_record = val_by_epoch.get(epoch, {})
        for key in val_keys:
            if is_number(val_record.get(key)):
                row[f"val/{key}"] = float(val_record[key])
        rows.append(row)

    fieldnames = ["epoch"]
    preferred = [
        "train/loss",
        "train/lr",
        "train/loss_rpn_cls",
        "train/loss_rpn_bbox",
        "train/s0.loss_cls",
        "train/s0.loss_bbox",
        "train/s0.loss_mask",
        "train/s1.loss_cls",
        "train/s1.loss_bbox",
        "train/s1.loss_mask",
        "train/s2.loss_cls",
        "train/s2.loss_bbox",
        "train/s2.loss_mask",
        "val/coco/segm_mAP_50",
        "val/coco/segm_mAP",
        "val/coco/segm_mAP_75",
        "val/coco/bbox_mAP_50",
        "val/coco/bbox_mAP",
        "val/coco/bbox_mAP_75",
        "val/coco/class1_precision",
        "val/coco/class2_precision",
        "val/coco/class3_precision",
        "val/coco/class4_precision",
    ]
    all_keys = sorted({key for row in rows for key in row.keys() if key != "epoch"})
    fieldnames.extend([key for key in preferred if key in all_keys])
    fieldnames.extend([key for key in all_keys if key not in fieldnames])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} epochs to {out_path}")


if __name__ == "__main__":
    main()

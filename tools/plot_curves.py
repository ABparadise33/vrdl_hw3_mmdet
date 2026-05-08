#!/usr/bin/env python3
"""Plot HW3 training loss and validation metric curves from a work_dir."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt


LOSS_KEYS = (
    "loss",
    "loss_rpn_cls",
    "loss_rpn_bbox",
    "loss_cls",
    "loss_bbox",
    "loss_mask",
)
METRIC_KEYS = (
    "coco/segm_mAP_50",
    "coco/bbox_mAP_50",
    "coco/segm_mAP",
    "coco/bbox_mAP",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work_dir", type=Path)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to write PNGs. Default: <work_dir>/curves.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, float]]:
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            parsed = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = value
            rows.append(parsed)
    return rows


def load_json_records(path: Path) -> List[Dict]:
    text = path.read_text().strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        if isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        pass

    records = []
    for line in text.splitlines():
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


def find_scalar_records(work_dir: Path) -> List[Dict]:
    records = []
    for path in sorted(work_dir.glob("**/vis_data/scalars.json")):
        records.extend(load_json_records(path))
    return records


def numeric(value) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def plot_lines(rows: List[Dict], x_key: str, y_keys: Iterable[str], out_path: Path, title: str) -> bool:
    plotted = False
    plt.figure(figsize=(10, 6))
    for key in y_keys:
        xs = []
        ys = []
        for idx, row in enumerate(rows):
            y = numeric(row.get(key))
            if y is None:
                continue
            x = numeric(row.get(x_key))
            xs.append(idx if x is None else x)
            ys.append(y)
        if xs:
            plt.plot(xs, ys, marker="o", linewidth=1.5, markersize=3, label=key)
            plotted = True
    if not plotted:
        plt.close()
        return False

    plt.title(title)
    plt.xlabel(x_key)
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return True


def write_records_csv(records: List[Dict], out_path: Path) -> None:
    if not records:
        return
    keys = sorted({key for record in records for key in record.keys()})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    args = parse_args()
    work_dir = args.work_dir
    out_dir = args.out_dir or work_dir / "curves"

    simple_train = work_dir / "simple_train_log.csv"
    simple_val = work_dir / "simple_val_log.csv"

    made_any = False
    if simple_train.exists():
        rows = read_csv(simple_train)
        made_any |= plot_lines(rows, "iter", ["avg_loss"], out_dir / "loss_curve.png", "Training Loss")
    if simple_val.exists():
        rows = read_csv(simple_val)
        made_any |= plot_lines(
            rows,
            "epoch",
            ["segm_mAP_50", "bbox_mAP_50", "segm_mAP", "bbox_mAP"],
            out_dir / "val_metrics_curve.png",
            "Validation Metrics",
        )

    scalar_records = find_scalar_records(work_dir)
    if scalar_records:
        write_records_csv(scalar_records, out_dir / "parsed_scalars.csv")
    if not simple_train.exists() and scalar_records:
        made_any |= plot_lines(
            scalar_records,
            "step",
            LOSS_KEYS,
            out_dir / "loss_curve.png",
            "Training Loss",
        )
    if not simple_val.exists() and scalar_records:
        made_any |= plot_lines(
            scalar_records,
            "step",
            METRIC_KEYS,
            out_dir / "val_metrics_curve.png",
            "Validation Metrics",
        )

    if not made_any:
        raise SystemExit(
            f"No plottable logs found under {work_dir}. "
            "Check for simple_train_log.csv or **/vis_data/scalars.json."
        )

    print(f"Wrote curves to {out_dir}")


if __name__ == "__main__":
    main()

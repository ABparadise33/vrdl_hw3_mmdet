#!/usr/bin/env python3
"""Collect logs and derived training artifacts into logs/ and results/."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exp_name")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Training work dir. Default: checkpoints/<exp_name>.",
    )
    return parser.parse_args()


def copy_logs(work_dir: Path, logs_dir: Path) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    patterns = [
        "*.log",
        "simple_train_log.csv",
        "simple_val_log.csv",
        "vis_data/scalars.json",
        "vis_data/config.py",
        "vis_data/*.json",
        "*/vis_data/scalars.json",
        "*/vis_data/config.py",
        "*/vis_data/*.json",
        "*.py",
    ]
    copied = 0
    for pattern in patterns:
        for src in work_dir.glob(pattern):
            if not src.is_file():
                continue
            rel = src.relative_to(work_dir)
            dst = logs_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
    print(f"Copied {copied} log/config files to {logs_dir}")


def run_tool(script: str, *args: str) -> None:
    subprocess.check_call([sys.executable, str(REPO_ROOT / "tools" / script), *args])


def main() -> None:
    args = parse_args()
    work_dir = args.work_dir or REPO_ROOT / "checkpoints" / args.exp_name
    results_dir = REPO_ROOT / "results" / args.exp_name
    logs_dir = REPO_ROOT / "logs" / args.exp_name

    if not work_dir.exists():
        raise SystemExit(f"work_dir does not exist: {work_dir}")

    results_dir.mkdir(parents=True, exist_ok=True)
    copy_logs(work_dir, logs_dir)
    run_tool("export_epoch_summary.py", str(work_dir), "--out", str(results_dir / "epoch_summary.csv"))
    run_tool("plot_curves.py", str(work_dir), "--out-dir", str(results_dir))
    print(f"Collected results under {results_dir}")


if __name__ == "__main__":
    main()

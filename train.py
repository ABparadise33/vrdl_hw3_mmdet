#!/usr/bin/env python3
"""Train HW3 Cascade Mask R-CNN from the repository root.

This entrypoint keeps the day-to-day command short:

    python train.py configs/cascade_mask_rcnn_r50_fpn_hw3.py --amp
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


# Keep the allocator conservative on RTX 4090/PyTorch 2.1.x. The
# expandable_segments option can trigger an internal CUDA allocator assert on
# some vast.ai images, so only cap split size here.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

from mmengine.config import Config, DictAction
from mmengine.runner import Runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to the MMDetection config.")
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Directory for logs and checkpoints.",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="Resume from the latest checkpoint, or from a given checkpoint path.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable automatic mixed precision training.",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Override MMEngine log level. Use WARNING for a quieter startup.",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        default={},
        help="Override config options, e.g. train_dataloader.batch_size=4.",
    )
    return parser.parse_args()


def enable_amp(cfg: Config) -> None:
    optim_wrapper = cfg.get("optim_wrapper", {})
    if optim_wrapper.get("type") == "AmpOptimWrapper":
        return
    optim_wrapper["type"] = "AmpOptimWrapper"
    optim_wrapper.setdefault("loss_scale", "dynamic")
    cfg.optim_wrapper = optim_wrapper


def apply_resume(cfg: Config, resume: str | None) -> None:
    if resume is None:
        return
    cfg.resume = True
    if resume != "auto":
        cfg.load_from = resume


def main() -> None:
    args = parse_args()
    cfg = Config.fromfile(args.config)

    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        cfg.work_dir = str(Path("work_dirs") / Path(args.config).stem)

    if args.log_level is not None:
        cfg.log_level = args.log_level
    if args.amp:
        enable_amp(cfg)
    apply_resume(cfg, args.resume)

    print(f"Config: {args.config}")
    print(f"Work dir: {cfg.work_dir}")
    print(f"Batch size: {cfg.train_dataloader.batch_size}")
    print(f"AMP: {'on' if args.amp else 'off'}")

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == "__main__":
    main()

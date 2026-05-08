#!/usr/bin/env python3
"""Train a detector with MMEngine Runner from a local config."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:128",
)

from mmengine.config import Config, DictAction
from mmengine.runner import Runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to MMDetection config.")
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Directory to save logs and checkpoints.",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="Resume from latest checkpoint, or from a given checkpoint path.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable automatic mixed precision training.",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        default={},
        help="Override config options, e.g. train_dataloader.batch_size=8.",
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
        config_stem = Path(args.config).stem
        cfg.work_dir = str(Path("work_dirs") / config_stem)

    if args.amp:
        enable_amp(cfg)
    apply_resume(cfg, args.resume)

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == "__main__":
    main()

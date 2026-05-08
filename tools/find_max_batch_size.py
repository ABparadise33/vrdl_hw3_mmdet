#!/usr/bin/env python3
"""Probe the largest train dataloader batch size that survives one iteration."""

from __future__ import annotations

import argparse
import copy
import os


os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:128",
)
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

import torch
from mmengine.config import Config
from mmengine.runner import Runner


def enable_amp(cfg: Config) -> None:
    optim_wrapper = cfg.get("optim_wrapper", {})
    if optim_wrapper.get("type") == "AmpOptimWrapper":
        return
    optim_wrapper["type"] = "AmpOptimWrapper"
    optim_wrapper.setdefault("loss_scale", "dynamic")
    cfg.optim_wrapper = optim_wrapper


def try_batch_size(config_path: str, batch_size: int, amp: bool) -> bool:
    cfg = Config.fromfile(config_path)
    cfg = copy.deepcopy(cfg)
    cfg.train_dataloader.batch_size = batch_size
    if amp:
        enable_amp(cfg)
    cfg.train_cfg = dict(type="IterBasedTrainLoop", max_iters=1, val_interval=999999)
    cfg.val_cfg = None
    cfg.val_dataloader = None
    cfg.val_evaluator = None
    cfg.default_hooks.checkpoint = dict(type="CheckpointHook", interval=999999)
    cfg.work_dir = f"/tmp/hw3_batch_probe_bs{batch_size}"
    runner = Runner.from_cfg(cfg)
    try:
        runner.train()
        torch.cuda.empty_cache()
        return True
    except RuntimeError as exc:
        torch.cuda.empty_cache()
        if "out of memory" in str(exc).lower():
            return False
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--max-batch", type=int, default=16)
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    best = 0
    for batch_size in range(args.start, args.max_batch + 1):
        print(f"Trying batch_size={batch_size}")
        if try_batch_size(args.config, batch_size, args.amp):
            best = batch_size
            print(f"OK: batch_size={batch_size}")
        else:
            print(f"OOM: batch_size={batch_size}")
            break
    print(f"Largest passing batch size: {best}")


if __name__ == "__main__":
    main()

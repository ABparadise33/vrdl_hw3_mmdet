#!/usr/bin/env python3
"""Probe the largest train dataloader batch size that survives one iteration."""

from __future__ import annotations

import argparse
import copy

import torch
from mmengine.config import Config
from mmengine.runner import Runner


def try_batch_size(config_path: str, batch_size: int) -> bool:
    cfg = Config.fromfile(config_path)
    cfg = copy.deepcopy(cfg)
    cfg.train_dataloader.batch_size = batch_size
    cfg.train_cfg = dict(type="IterBasedTrainLoop", max_iters=1, val_interval=1)
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    best = 0
    for batch_size in range(args.start, args.max_batch + 1):
        print(f"Trying batch_size={batch_size}")
        if try_batch_size(args.config, batch_size):
            best = batch_size
            print(f"OK: batch_size={batch_size}")
        else:
            print(f"OOM: batch_size={batch_size}")
            break
    print(f"Largest passing batch size: {best}")


if __name__ == "__main__":
    main()


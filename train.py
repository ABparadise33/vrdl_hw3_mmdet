#!/usr/bin/env python3
"""Train HW3 Cascade Mask R-CNN from the repository root.

This entrypoint keeps the day-to-day command short:

    python train.py configs/cascade_mask_rcnn_r50_fpn_hw3.py --amp
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from numbers import Number
import os
import sys
from pathlib import Path


# Keep the allocator conservative on RTX 4090/PyTorch 2.1.x. The
# expandable_segments option can trigger an internal CUDA allocator assert on
# some vast.ai images, so only cap split size here.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

from mmengine.config import Config, DictAction
from mmengine.hooks import Hook
from mmengine.runner import Runner


class TeeStream:
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to the MMDetection config.")
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Directory for logs and checkpoints.",
    )
    parser.add_argument(
        "--exp-name",
        default=None,
        help="Experiment name. If --work-dir is omitted, checkpoints go to checkpoints/<exp-name>.",
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
        "--print-interval",
        type=int,
        default=5,
        help="Print concise training progress every N iterations.",
    )
    parser.add_argument(
        "--no-simple-progress",
        action="store_true",
        help="Disable the concise HW3 progress logger.",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        default={},
        help="Override config options, e.g. train_dataloader.batch_size=4.",
    )
    return parser.parse_args()


def setup_terminal_log(exp_name: str | None):
    if exp_name is None:
        return None
    logs_dir = Path("logs") / exp_name
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = (logs_dir / "terminal.log").open("a", buffering=1)
    sys.stdout = TeeStream(sys.stdout, log_file)
    sys.stderr = TeeStream(sys.stderr, log_file)
    return log_file


def to_float(value) -> float | None:
    if hasattr(value, "detach"):
        return float(value.detach().mean().cpu())
    if isinstance(value, Number):
        return float(value)
    return None


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    return f"{minutes:d}m{secs:02d}s"


def dataloader_batch_size(dataloader) -> int | str:
    for obj in (dataloader, getattr(dataloader, "batch_sampler", None)):
        batch_size = getattr(obj, "batch_size", None)
        if batch_size is not None:
            return batch_size
    return "?"


class SimpleProgressHook(Hook):
    """Small HW2-style progress lines for long MMDetection runs."""

    def __init__(self, interval: int = 5) -> None:
        self.interval = max(1, interval)
        self.loss_sums = defaultdict(float)
        self.loss_count = 0
        self.epoch_start = 0.0
        self.epoch_iters = 0
        self.train_csv: Path | None = None
        self.val_csv: Path | None = None

    def before_run(self, runner) -> None:
        work_dir = Path(runner.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        self.train_csv = work_dir / "simple_train_log.csv"
        self.val_csv = work_dir / "simple_val_log.csv"
        if not self.train_csv.exists():
            self.train_csv.write_text("epoch,iter,total_iters,avg_loss,lr,iter_time,eta_seconds\n")
        if not self.val_csv.exists():
            self.val_csv.write_text("epoch,bbox_mAP_50,segm_mAP_50,bbox_mAP,segm_mAP\n")

    def before_train_epoch(self, runner) -> None:
        self.loss_sums.clear()
        self.loss_count = 0
        self.epoch_start = time.time()
        self.epoch_iters = len(runner.train_dataloader)
        max_epochs = getattr(runner.train_loop, "max_epochs", "?")
        print(
            f"[train] epoch {runner.epoch + 1}/{max_epochs} "
            f"images={len(runner.train_dataloader.dataset)} "
            f"iters={self.epoch_iters} "
            f"batch_size={dataloader_batch_size(runner.train_dataloader)}",
            flush=True,
        )

    def after_train_iter(self, runner, batch_idx: int, data_batch=None, outputs=None) -> None:
        losses = {}
        if isinstance(outputs, dict):
            for key, value in outputs.items():
                if "loss" not in key:
                    continue
                scalar = to_float(value)
                if scalar is not None:
                    losses[key] = scalar

        if losses:
            self.loss_count += 1
            for key, value in losses.items():
                self.loss_sums[key] += value

        step = batch_idx + 1
        if step % self.interval != 0 and step != self.epoch_iters:
            return

        avg_loss = self.average_loss()
        elapsed = time.time() - self.epoch_start
        iter_time = elapsed / max(1, step)
        eta = iter_time * max(0, self.epoch_iters - step)
        lr = self.current_lr(runner)
        max_epochs = getattr(runner.train_loop, "max_epochs", "?")
        eta_seconds = int(max(0, eta))
        print(
            f"[train] epoch {runner.epoch + 1}/{max_epochs} "
            f"iter {step}/{self.epoch_iters} "
            f"avg_loss={avg_loss:.4f} "
            f"lr={lr:.2e} "
            f"iter_time={iter_time:.2f}s "
            f"eta={format_seconds(eta_seconds)}",
            flush=True,
        )
        if self.train_csv is not None:
            with self.train_csv.open("a") as f:
                f.write(
                    f"{runner.epoch + 1},{step},{self.epoch_iters},"
                    f"{avg_loss:.6f},{lr:.8g},{iter_time:.4f},{eta_seconds}\n"
                )

    def after_val_epoch(self, runner, metrics=None) -> None:
        if not metrics:
            return
        wanted = [
            ("bbox50", "coco/bbox_mAP_50"),
            ("segm50", "coco/segm_mAP_50"),
            ("bbox", "coco/bbox_mAP"),
            ("segm", "coco/segm_mAP"),
        ]
        parts = []
        for label, key in wanted:
            if key in metrics:
                parts.append(f"{label}={metrics[key]:.4f}")
        if parts:
            print(f"[val] epoch {runner.epoch} " + " ".join(parts), flush=True)
        if self.val_csv is not None:
            with self.val_csv.open("a") as f:
                f.write(
                    f"{runner.epoch},"
                    f"{metrics.get('coco/bbox_mAP_50', 0.0):.6f},"
                    f"{metrics.get('coco/segm_mAP_50', 0.0):.6f},"
                    f"{metrics.get('coco/bbox_mAP', 0.0):.6f},"
                    f"{metrics.get('coco/segm_mAP', 0.0):.6f}\n"
                )

    def average_loss(self) -> float:
        if self.loss_count == 0:
            return 0.0
        if "loss" in self.loss_sums:
            return self.loss_sums["loss"] / self.loss_count
        return sum(self.loss_sums.values()) / self.loss_count

    @staticmethod
    def current_lr(runner) -> float:
        try:
            lr_info = runner.optim_wrapper.get_lr()
        except Exception:
            return 0.0
        if isinstance(lr_info, dict):
            for value in lr_info.values():
                if isinstance(value, (list, tuple)) and value:
                    return float(value[0])
                if isinstance(value, Number):
                    return float(value)
        return 0.0


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
    setup_terminal_log(args.exp_name)
    cfg = Config.fromfile(args.config)

    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif args.exp_name is not None:
        cfg.work_dir = str(Path("checkpoints") / args.exp_name)
    elif cfg.get("work_dir", None) is None:
        cfg.work_dir = str(Path("work_dirs") / Path(args.config).stem)

    if args.log_level is not None:
        cfg.log_level = args.log_level
    if args.amp:
        enable_amp(cfg)
    apply_resume(cfg, args.resume)
    if not args.no_simple_progress:
        cfg.default_hooks.logger.interval = 1000000

    print(f"Config: {args.config}")
    print(f"Work dir: {cfg.work_dir}")
    if args.exp_name:
        print(f"Logs dir: logs/{args.exp_name} (run tools/collect_results.py after training)")
        print(f"Results dir: results/{args.exp_name}")
    print(f"Batch size: {cfg.train_dataloader.batch_size}")
    print(f"AMP: {'on' if args.amp else 'off'}")

    runner = Runner.from_cfg(cfg)
    if not args.no_simple_progress:
        runner.register_hook(SimpleProgressHook(args.print_interval), priority="LOW")
    runner.train()


if __name__ == "__main__":
    main()

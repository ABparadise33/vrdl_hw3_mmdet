#!/usr/bin/env python3
"""Run MMDetection inference and write an HW3 CodaBench submission."""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORE_THRESHOLDS = {
    1: 0.35,
    2: 0.35,
    3: 0.25,
    4: 0.25,
}
DEFAULT_AREA_RANGES = {
    1: (20, 10000),
    2: (10, 3000),
    3: (80, 6000),
    4: (120, 90000),
}


def encode_mask(binary_mask: np.ndarray) -> Dict:
    rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def xyxy_to_xywh(box: np.ndarray) -> List[float]:
    x1, y1, x2, y2 = box[:4].tolist()
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def load_detector(config: str, checkpoint: str, device: str):
    from mmdet.apis import init_detector

    return init_detector(config, checkpoint, device=device)


def load_tta_detector(config: str, checkpoint: str, device: str):
    """Build a DetTTAModel without using init_detector's backbone shortcut.

    MMDetection's init_detector assumes cfg.model has a top-level backbone.
    DetTTAModel stores the real detector under cfg.model.module, so building the
    wrapper there triggers an AttributeError. Building/loading the base detector
    first and then wrapping it keeps TTA compatible with the inference API.
    """

    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.runner import load_checkpoint
    from mmdet.registry import MODELS

    cfg = Config.fromfile(config)
    init_default_scope(cfg.get("default_scope", "mmdet"))

    base_cfg = cfg.model.copy()
    base_cfg.train_cfg = None
    base_model = MODELS.build(base_cfg)
    checkpoint_data = load_checkpoint(base_model, checkpoint, map_location="cpu")

    checkpoint_meta = checkpoint_data.get("meta", {})
    if "dataset_meta" in checkpoint_meta:
        base_model.dataset_meta = checkpoint_meta["dataset_meta"]
    else:
        base_model.dataset_meta = {"classes": cfg.get("classes", None)}

    tta_cfg = cfg.tta_model.copy()
    tta_cfg.module = base_model
    model = MODELS.build(tta_cfg)
    model.dataset_meta = base_model.dataset_meta
    cfg.test_pipeline = cfg.tta_pipeline
    cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
    model.cfg = cfg
    model.to(device)
    model.eval()
    return model


def run_inference(model, image_path: Path):
    from mmdet.apis import inference_detector

    return inference_detector(model, str(image_path))


def instances_from_result(result):
    pred = result.pred_instances.cpu()
    boxes = pred.bboxes.numpy()
    scores = pred.scores.numpy()
    labels = pred.labels.numpy()
    masks = pred.masks.numpy() if hasattr(pred, "masks") else None
    return boxes, scores, labels, masks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--test-dir", type=Path, default=REPO_ROOT / "data/test_release")
    parser.add_argument(
        "--mapping",
        type=Path,
        default=REPO_ROOT / "data/test_image_name_to_ids.json",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=REPO_ROOT / "submissions/test-results.json",
    )
    parser.add_argument(
        "--out-zip",
        type=Path,
        default=REPO_ROOT / "submissions/test-results.zip",
    )
    parser.add_argument(
        "--score-thr",
        type=float,
        default=None,
        help=(
            "Global score threshold. Default is 0.05 without --adaptive; "
            "with --adaptive, omitted means class-wise thresholds."
        ),
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Enable class-wise score thresholds and class-wise area filtering.",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Use the config's TTA model/pipeline if supported by installed MMDetection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        "Inference mode: "
        f"TTA={'on' if args.tta else 'off'}, "
        f"adaptive={'on' if args.adaptive else 'off'}"
    )
    if args.tta:
        model = load_tta_detector(args.config, args.checkpoint, args.device)
    else:
        model = load_detector(args.config, args.checkpoint, args.device)

    image_infos = json.loads(args.mapping.read_text())
    results = []
    for info in tqdm(image_infos, desc="Infer test"):
        image_path = args.test_dir / info["file_name"]
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"Failed to read {image_path}")
        result = run_inference(model, image_path)
        boxes, scores, labels, masks = instances_from_result(result)
        if masks is None:
            continue

        for box, score, label, mask in zip(boxes, scores, labels, masks):
            category_id = int(label) + 1
            threshold = args.score_thr
            if args.adaptive and threshold is None:
                threshold = DEFAULT_SCORE_THRESHOLDS[category_id]
            elif threshold is None:
                threshold = 0.05
            if float(score) < threshold:
                continue
            binary_mask = mask.astype(bool)
            if args.adaptive:
                area = int(binary_mask.sum())
                area_min, area_max = DEFAULT_AREA_RANGES[category_id]
                if area < area_min or area > area_max:
                    continue
            results.append(
                {
                    "image_id": int(info["id"]),
                    "bbox": xyxy_to_xywh(box),
                    "score": float(score),
                    "category_id": category_id,
                    "segmentation": encode_mask(binary_mask),
                }
            )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(results))
    args.out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(args.out_json, arcname="test-results.json")
    print(f"Wrote {len(results)} predictions to {args.out_json}")
    print(f"Wrote zip submission to {args.out_zip} as test-results.json")


if __name__ == "__main__":
    main()

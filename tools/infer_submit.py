#!/usr/bin/env python3
"""Run MMDetection inference and write an HW3 CodaBench submission."""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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


def run_inference(model, image_path: Path):
    from mmdet.apis import inference_detector

    if isinstance(image_path, Path):
        image_path = str(image_path)
    return inference_detector(model, image_path)


def instances_from_result(result):
    pred = result.pred_instances.cpu()
    boxes = pred.bboxes.numpy()
    scores = pred.scores.numpy()
    labels = pred.labels.numpy()
    masks = pred.masks.numpy() if hasattr(pred, "masks") else None
    return boxes, scores, labels, masks


def resize_keep_ratio(image: np.ndarray, target_size: int) -> Tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = min(target_size / max(height, 1), target_size / max(width, 1))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return resized, scale


def resize_masks_to_original(masks: np.ndarray, original_shape: Tuple[int, int]) -> np.ndarray:
    original_height, original_width = original_shape
    resized_masks = []
    for mask in masks:
        resized = cv2.resize(
            mask.astype(np.uint8),
            (original_width, original_height),
            interpolation=cv2.INTER_NEAREST,
        )
        resized_masks.append(resized.astype(bool))
    if not resized_masks:
        return np.zeros((0, original_height, original_width), dtype=bool)
    return np.stack(resized_masks, axis=0)


def nms_instances(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    masks: np.ndarray,
    iou_threshold: float,
    max_per_img: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(boxes) == 0:
        return boxes, scores, labels, masks

    import torch
    from torchvision.ops import nms

    keep_indices = []
    for label in np.unique(labels):
        cls_indices = np.where(labels == label)[0]
        cls_keep = nms(
            torch.as_tensor(boxes[cls_indices], dtype=torch.float32),
            torch.as_tensor(scores[cls_indices], dtype=torch.float32),
            iou_threshold,
        )
        keep_indices.extend(cls_indices[cls_keep.cpu().numpy()].tolist())

    keep_indices = sorted(keep_indices, key=lambda idx: float(scores[idx]), reverse=True)
    keep_indices = keep_indices[:max_per_img]
    return boxes[keep_indices], scores[keep_indices], labels[keep_indices], masks[keep_indices]


def run_manual_tta(
    model,
    image: np.ndarray,
    scales: Sequence[int],
    use_flip: bool,
    nms_iou: float,
    max_per_img: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    original_height, original_width = image.shape[:2]
    all_boxes = []
    all_scores = []
    all_labels = []
    all_masks = []

    for scale_size in scales:
        resized_image, scale_factor = resize_keep_ratio(image, scale_size)
        _, resized_width = resized_image.shape[:2]
        flip_options = (False, True) if use_flip else (False,)

        for flip in flip_options:
            aug_image = cv2.flip(resized_image, 1) if flip else resized_image
            result = run_inference(model, aug_image)
            boxes, scores, labels, masks = instances_from_result(result)
            if masks is None or len(boxes) == 0:
                continue

            if flip:
                x1 = boxes[:, 0].copy()
                x2 = boxes[:, 2].copy()
                boxes[:, 0] = resized_width - x2
                boxes[:, 2] = resized_width - x1
                masks = np.ascontiguousarray(np.flip(masks, axis=2))

            boxes = boxes / scale_factor
            boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, original_width)
            boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, original_height)
            masks = resize_masks_to_original(masks, (original_height, original_width))

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
            all_masks.append(masks)

    if not all_boxes:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, original_height, original_width), dtype=bool),
        )

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    masks = np.concatenate(all_masks, axis=0)
    return nms_instances(boxes, scores, labels, masks, nms_iou, max_per_img)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument(
        "--exp-name",
        default=None,
        help="Experiment name. If set, outputs default to results/<exp-name>/.",
    )
    parser.add_argument(
        "--result-name",
        default=None,
        help="Output stem under results/<exp-name>/ when --exp-name is used.",
    )
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
        default=None,
    )
    parser.add_argument(
        "--out-zip",
        type=Path,
        default=None,
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
        help="Use manual multi-scale/flip TTA for instance segmentation.",
    )
    parser.add_argument(
        "--tta-scales",
        nargs="+",
        type=int,
        default=[800, 1000, 1200],
        help="Square max-side sizes for manual TTA.",
    )
    parser.add_argument(
        "--tta-no-flip",
        action="store_true",
        help="Disable horizontal flip inside manual TTA.",
    )
    parser.add_argument(
        "--tta-nms-iou",
        type=float,
        default=0.5,
        help="Per-class bbox NMS IoU for merging manual TTA predictions.",
    )
    return parser.parse_args()


def default_result_name(args: argparse.Namespace) -> str:
    parts = []
    parts.append("tta" if args.tta else "no_tta")
    parts.append("adaptive" if args.adaptive else "no_adaptive")
    return "_".join(parts)


def resolve_output_paths(args: argparse.Namespace) -> None:
    if args.out_json is not None and args.out_zip is not None:
        return
    if args.exp_name:
        out_dir = REPO_ROOT / "results" / args.exp_name
        stem = args.result_name or default_result_name(args)
    else:
        out_dir = REPO_ROOT / "results"
        stem = args.result_name or "test-results"
    if args.out_json is None:
        args.out_json = out_dir / f"{stem}.json"
    if args.out_zip is None:
        args.out_zip = out_dir / f"{stem}.zip"


def load_image_infos(mapping_path: Path) -> List[Dict]:
    data = json.loads(mapping_path.read_text())
    if isinstance(data, dict) and "images" in data:
        return data["images"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported mapping format: {mapping_path}")


def main() -> None:
    args = parse_args()
    resolve_output_paths(args)
    print(
        "Inference mode: "
        f"TTA={'on' if args.tta else 'off'}, "
        f"adaptive={'on' if args.adaptive else 'off'}"
    )
    model = load_detector(args.config, args.checkpoint, args.device)

    image_infos = load_image_infos(args.mapping)
    results = []
    for info in tqdm(image_infos, desc="Infer test"):
        image_path = args.test_dir / info["file_name"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read {image_path}")
        if args.tta:
            boxes, scores, labels, masks = run_manual_tta(
                model,
                image,
                scales=args.tta_scales,
                use_flip=not args.tta_no_flip,
                nms_iou=args.tta_nms_iou,
                max_per_img=300,
            )
        else:
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

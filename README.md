# VRDL HW3 Cascade Mask R-CNN

This repository contains the MMDetection-based training pipeline for HW3
instance segmentation.

## Why MMDetection

The original Cascade R-CNN repositories are useful references, but the Detectron
implementation depends on Python 2 and Caffe2-era CUDA. For training on a 4090,
we use MMDetection with Cascade Mask R-CNN + ResNet-50-FPN and COCO pretrained
weights.

## Setup

Recommended environment on vast.ai:

```bash
conda create -n hw3 python=3.10 -y
conda activate hw3
pip install -U pip
pip install -r requirements.txt
mim install mmengine mmcv mmdet
```

If CUDA/PyTorch are not already available in the image, install the PyTorch
build that matches the CUDA version first, then install MMDetection.

## Download Dataset

The dataset is hosted on Google Drive. From the repository root:

```bash
python tools/download_dataset.py
```

Expected layout after extraction:

```text
data/
  train/
  test_release/
  test_image_name_to_ids.json
```

## Prepare COCO Annotations

From the repository root:

```bash
python tools/convert_hw3_to_coco.py
```

This reads `data/train`, writes COCO JSON files to `annotations`, and
does not modify the original dataset.

Generated files:

```text
annotations/instances_hw3_train.json
annotations/instances_hw3_val.json
annotations/image_info_hw3_test.json
annotations/split_hw3.json
```

## Train

From an MMDetection repository root or an environment where `mim` can run
MMDetection tools:

```bash
mim train mmdet configs/cascade_mask_rcnn_r50_fpn_hw3.py \
  --work-dir work_dirs/cascade_mask_rcnn_r50_fpn_hw3
```

The config uses:

- `cascade_mask_rcnn_r50_fpn_1x_coco` as the base architecture.
- COCO pretrained checkpoint via `load_from`.
- 4 HW3 foreground classes.
- Validation by COCO bbox/mask metrics.
- Epoch-wise checkpointing and metrics logging.

## Find Max Batch Size

Run this on the 4090 after MMDetection is installed:

```bash
python tools/find_max_batch_size.py \
  configs/cascade_mask_rcnn_r50_fpn_hw3.py \
  --start 1 --max-batch 16
```

Use the largest passing batch size with a safety margin, or use gradient
accumulation if you want a larger effective batch size.

## Inference and Submission

```bash
python tools/infer_submit.py \
  configs/cascade_mask_rcnn_r50_fpn_hw3.py \
  work_dirs/cascade_mask_rcnn_r50_fpn_hw3/best_coco_segm_mAP_50_epoch_*.pth \
  --out-json submissions/test-results.json \
  --out-zip submissions/test-results.zip
```

The script supports class-wise score thresholds, area filtering, and compressed
COCO RLE output.

## Full vast.ai Terminal Sequence

```bash
git clone <YOUR_GITHUB_REPO_URL>
cd <YOUR_REPO_NAME>

conda create -n hw3 python=3.10 -y
conda activate hw3
pip install -U pip
pip install -r requirements.txt
mim install mmengine mmcv mmdet

python tools/download_dataset.py
python tools/convert_hw3_to_coco.py

python tools/find_max_batch_size.py configs/cascade_mask_rcnn_r50_fpn_hw3.py \
  --start 1 --max-batch 16

mim train mmdet configs/cascade_mask_rcnn_r50_fpn_hw3.py \
  --work-dir work_dirs/cascade_mask_rcnn_r50_fpn_hw3

python tools/infer_submit.py \
  configs/cascade_mask_rcnn_r50_fpn_hw3.py \
  work_dirs/cascade_mask_rcnn_r50_fpn_hw3/best_coco_segm_mAP_50_epoch_*.pth \
  --tta
```

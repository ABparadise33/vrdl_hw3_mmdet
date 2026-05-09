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

# Install PyTorch first. MIM needs torch to detect CUDA and select mmcv wheels.
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
pip install "numpy<2" --force-reinstall
mim install "mmengine>=0.7.1" "mmcv==2.1.0" "mmdet==3.3.0"

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY
```

If your vast.ai image already has a working PyTorch install, you can skip the
PyTorch install line. If the CUDA version is different, use the matching command
from the PyTorch install selector.

The rotate90 augmentation uses MMDetection's `Albu` wrapper. Keep
`albumentations==1.3.1`; newer Albumentations versions can raise
`ValueError: Key img_path is not in available keys`.

If reinstalling Albumentations upgrades NumPy or packaging, repair the
environment with:

```bash
pip install "numpy<2" "packaging~=24.0" --force-reinstall
pip install albumentations==1.3.1 --no-deps
```

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

For the AdamW/cosine baseline that follows the classmate-style training split,
regenerate annotations with an 85/15 train/val split:

```bash
python tools/convert_hw3_to_coco.py \
  --val-ratio 0.15 \
  --out-dir annotations_adamw85
```

For the normalized PNG experiment, export p1-p99 normalized 3-channel PNGs and
matching annotations without modifying the original `data/` folder:

```bash
python tools/convert_hw3_to_coco.py \
  --val-ratio 0.15 \
  --out-dir annotations_norm_png \
  --export-images-dir data_norm_png \
  --percentile-normalize
```

## Train

Run training from the repository root:

```bash
python train.py configs/cascade_mask_rcnn_r50_fpn_hw3.py \
  --exp-name baseline_bs4_amp \
  --amp \
  --cfg-options train_dataloader.batch_size=4
```

With `--exp-name`, checkpoints are written to `checkpoints/<exp_name>`.
After training, collect logs and derived artifacts:

```bash
python tools/collect_results.py baseline_bs4_amp
```

This writes:

```text
checkpoints/<exp_name>/*.pth
logs/<exp_name>/
results/<exp_name>/epoch_summary.csv
results/<exp_name>/loss_curve.png
results/<exp_name>/val_metrics_curve.png
```

The config uses:

- `cascade_mask_rcnn_r50_fpn_1x_coco` as the base architecture.
- COCO pretrained checkpoint via `load_from`.
- 4 HW3 foreground classes.
- Validation by COCO bbox/mask metrics.
- Epoch-wise checkpointing and metrics logging.

`train.py` adds a concise progress line every 5 iterations by default:

```text
[train] epoch 2/24 iter 35/84 avg_loss=1.2345 lr=2.50e-03 iter_time=1.80s eta=1m28s
```

Change the frequency with `--print-interval 1`, or disable it with
`--no-simple-progress`.

The same concise values are saved for plotting:

```text
work_dirs/<run_name>/simple_train_log.csv
work_dirs/<run_name>/simple_val_log.csv
```

For older runs that only have MMEngine logs, plot curves from either source:

```bash
python tools/plot_curves.py work_dirs/cascade_mask_rcnn_r50_fpn_hw3_bs4_amp
```

Outputs:

```text
work_dirs/<run_name>/curves/loss_curve.png
work_dirs/<run_name>/curves/val_metrics_curve.png
work_dirs/<run_name>/curves/parsed_scalars.csv
```

Export one row per epoch with averaged train losses and validation AP metrics:

```bash
python tools/export_epoch_summary.py work_dirs/cascade_mask_rcnn_r50_fpn_hw3_bs4_amp
```

Output:

```text
work_dirs/<run_name>/epoch_summary.csv
```

For the new folder layout, prefer:

```bash
python tools/collect_results.py <exp_name>
```

MMEngine prints the system environment, resolved config, and hook order at
startup. That wall of text is normal and only happens before training begins.
If you want a quieter start, add `--log-level WARNING`.

For RTX 4090, start with AMP and `train_dataloader.batch_size=2`. If the
batch-size probe passes cleanly, try 4 next. The scripts set
`PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128` by default to reduce memory
fragmentation without using PyTorch's fragile `expandable_segments` mode.

## Find Max Batch Size

Run this on the 4090 after MMDetection is installed:

```bash
python tools/find_max_batch_size.py \
  configs/cascade_mask_rcnn_r50_fpn_hw3.py \
  --start 1 --max-batch 16 --amp
```

Use the largest passing batch size with a safety margin, or use gradient
accumulation if you want a larger effective batch size.

## Augmentation Experiments

Run one experiment at a time so the report can compare each added technique
against the baseline.

Classmate-style training baseline: AdamW, cosine LR schedule, 36 epochs. Use
the 85/15 annotations generated above.

```bash
python train.py configs/cascade_mask_rcnn_r50_fpn_hw3_adamw_cosine_85split.py \
  --exp-name baseline_adamw_cosine_85split_bs4_amp \
  --amp \
  --cfg-options train_dataloader.batch_size=4
```

Exp1: p1-p99 normalized 3-channel PNGs with the same AdamW/cosine schedule.

```bash
python train.py configs/cascade_mask_rcnn_r50_fpn_hw3_adamw_cosine_norm_png.py \
  --exp-name exp1_norm_png_adamw_cosine_bs4_amp \
  --amp \
  --cfg-options train_dataloader.batch_size=4
```

Exp2: normalized PNGs plus horizontal flip, vertical flip, and color
augmentation.

```bash
python train.py configs/cascade_mask_rcnn_r50_fpn_hw3_adamw_cosine_norm_png_aug.py \
  --exp-name exp2_norm_png_flip_color_bs4_amp \
  --amp \
  --cfg-options train_dataloader.batch_size=4
```

```bash
python train.py configs/cascade_mask_rcnn_r50_fpn_hw3_vflip.py \
  --exp-name exp_vflip_bs4_amp \
  --amp \
  --cfg-options train_dataloader.batch_size=4

python train.py configs/cascade_mask_rcnn_r50_fpn_hw3_rotate90.py \
  --exp-name exp_rotate90_bs4_amp \
  --amp \
  --cfg-options train_dataloader.batch_size=4

python train.py configs/cascade_mask_rcnn_r50_fpn_hw3_photometric.py \
  --exp-name exp_photometric_bs4_amp \
  --amp \
  --cfg-options train_dataloader.batch_size=4

python train.py configs/cascade_mask_rcnn_r50_fpn_hw3_multiscale.py \
  --exp-name exp_multiscale_bs4_amp \
  --amp \
  --cfg-options train_dataloader.batch_size=4

python train.py configs/cascade_mask_rcnn_r50_fpn_hw3_flip_photometric.py \
  --exp-name exp_flip_photometric_bs4_amp \
  --amp \
  --cfg-options train_dataloader.batch_size=4
```

## Inference and Submission

Baseline inference without TTA or adaptive post-processing:

```bash
python inference.py \
  configs/cascade_mask_rcnn_r50_fpn_hw3.py \
  checkpoints/baseline_bs4_amp/best_coco_segm_mAP_50_epoch_*.pth \
  --exp-name baseline_bs4_amp \
  --result-name best_no_tta_no_adaptive
```

TTA + adaptive post-processing inference:

```bash
python inference.py \
  configs/cascade_mask_rcnn_r50_fpn_hw3.py \
  checkpoints/baseline_bs4_amp/best_coco_segm_mAP_50_epoch_*.pth \
  --exp-name baseline_bs4_amp \
  --result-name best_tta_adaptive \
  --tta \
  --adaptive
```

Adaptive mode enables class-wise score thresholds and class-wise area filtering.
The `--tta` mode uses a manual instance-segmentation TTA path instead of
MMDetection's `DetTTAModel`, because MMDetection 3.3.0 asserts that mask TTA is
not supported by the built-in wrapper. By default it runs scales
`800 1000 1200` with horizontal flip and merges predictions with per-class NMS.
Both modes write compressed COCO RLE submission files. The zip filename can be
experiment-specific, but the JSON inside the zip is always named
`test-results.json` for CodaBench submission.

For validation-set confusion matrix, first write validation predictions, then
plot the matrix:

```bash
python inference.py \
  configs/cascade_mask_rcnn_r50_fpn_hw3.py \
  checkpoints/baseline_bs4_amp/epoch_24.pth \
  --mapping annotations/instances_hw3_val.json \
  --test-dir data/train \
  --exp-name baseline_bs4_amp \
  --result-name val_epoch24_no_tta_no_adaptive

python tools/plot_confusion_matrix.py \
  annotations/instances_hw3_val.json \
  results/baseline_bs4_amp/val_epoch24_no_tta_no_adaptive.json \
  --out-dir results/baseline_bs4_amp
```

Sweep validation score thresholds:

```bash
python tools/sweep_thresholds.py \
  configs/cascade_mask_rcnn_r50_fpn_hw3.py \
  checkpoints/baseline_bs4_amp/epoch_24.pth \
  --exp-name baseline_bs4_amp \
  --thresholds 0.01 0.03 0.05 0.08 0.10 0.15 0.20 0.25 0.30 0.35 0.40
```

Output:

```text
results/<exp_name>/threshold_sweep.csv
```

## Full vast.ai Terminal Sequence

```bash
git clone <YOUR_GITHUB_REPO_URL>
cd <YOUR_REPO_NAME>

conda create -n hw3 python=3.10 -y
conda activate hw3
pip install -U pip
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install "numpy<2" --force-reinstall
mim install "mmengine>=0.7.1" "mmcv==2.1.0" "mmdet==3.3.0"

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY

python tools/download_dataset.py
python tools/convert_hw3_to_coco.py

python tools/find_max_batch_size.py configs/cascade_mask_rcnn_r50_fpn_hw3.py \
  --start 1 --max-batch 16 --amp

python train.py configs/cascade_mask_rcnn_r50_fpn_hw3.py \
  --work-dir work_dirs/cascade_mask_rcnn_r50_fpn_hw3_bs2_amp \
  --amp \
  --cfg-options train_dataloader.batch_size=2

python inference.py \
  configs/cascade_mask_rcnn_r50_fpn_hw3.py \
  work_dirs/cascade_mask_rcnn_r50_fpn_hw3/best_coco_segm_mAP_50_epoch_*.pth \
  --tta \
  --adaptive
```

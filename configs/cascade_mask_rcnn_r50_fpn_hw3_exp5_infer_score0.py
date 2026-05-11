"""Exp5 inference-only config with internal score_thr=0.0.

Use this with checkpoints trained from:
  cascade_mask_rcnn_r50_fpn_hw3_exp4_rle_aug_loss125_multiscale.py

This does not change training. It only prevents MMDetection from dropping
low-score detections before our submission/threshold-sweep scripts can decide
which scores to keep.
"""

_base_ = "./cascade_mask_rcnn_r50_fpn_hw3_exp4_rle_aug_loss125_multiscale.py"

model = dict(
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type="nms", iou_threshold=0.7),
            min_bbox_size=0,
        ),
        rcnn=dict(
            score_thr=0.0,
            nms=dict(type="nms", iou_threshold=0.5),
            max_per_img=300,
            mask_thr_binary=0.5,
        ),
    )
)

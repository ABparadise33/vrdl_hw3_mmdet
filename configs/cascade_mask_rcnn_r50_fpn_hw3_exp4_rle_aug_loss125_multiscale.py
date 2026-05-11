"""Exp4 multi-scale: Exp4 baseline plus random multi-scale training."""

_base_ = "./cascade_mask_rcnn_r50_fpn_hw3_exp4_rle_aug_loss125.py"

train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(
        type="RandomChoiceResize",
        scales=[
            (768, 768),
            (896, 896),
            (1024, 1024),
            (1152, 1152),
            (1280, 1280),
        ],
        keep_ratio=True,
    ),
    dict(type="RandomFlip", prob=0.5, direction="horizontal"),
    dict(type="RandomFlip", prob=0.5, direction="vertical"),
    dict(
        type="PhotoMetricDistortion",
        brightness_delta=24,
        contrast_range=(0.8, 1.25),
        saturation_range=(0.8, 1.25),
        hue_delta=12,
    ),
    dict(type="PackDetInputs"),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

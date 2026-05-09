"""HW3 experiment: horizontal/vertical flip + brightness/contrast/color."""

_base_ = "./cascade_mask_rcnn_r50_fpn_hw3.py"

train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(type="Resize", scale=(1333, 800), keep_ratio=True),
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

"""HW3 experiment: baseline + multi-scale training."""

_base_ = "./cascade_mask_rcnn_r50_fpn_hw3.py"

train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(
        type="RandomChoiceResize",
        scales=[
            (1000, 600),
            (1200, 720),
            (1333, 800),
            (1500, 900),
        ],
        keep_ratio=True,
    ),
    dict(type="RandomFlip", prob=0.5, direction="horizontal"),
    dict(type="PackDetInputs"),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

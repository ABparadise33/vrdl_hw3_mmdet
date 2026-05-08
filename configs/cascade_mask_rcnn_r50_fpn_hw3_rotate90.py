"""HW3 experiment: baseline + random 90-degree rotation augmentation."""

_base_ = "./cascade_mask_rcnn_r50_fpn_hw3.py"

train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(type="Resize", scale=(1333, 800), keep_ratio=True),
    dict(type="RandomFlip", prob=0.5, direction="horizontal"),
    dict(
        type="Albu",
        transforms=[
            dict(type="RandomRotate90", p=0.75),
        ],
        bbox_params=dict(
            type="BboxParams",
            format="pascal_voc",
            label_fields=["gt_bboxes_labels", "gt_ignore_flags"],
            min_visibility=0.0,
            filter_lost_elements=True,
        ),
        keymap={
            "img": "image",
            "gt_masks": "masks",
            "gt_bboxes": "bboxes",
        },
        skip_img_without_anno=True,
    ),
    dict(type="PackDetInputs"),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

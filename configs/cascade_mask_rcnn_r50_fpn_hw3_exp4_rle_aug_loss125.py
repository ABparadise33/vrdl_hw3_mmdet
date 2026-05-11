"""Exp4 baseline: classmate-style RLE data + our best aug/loss settings.

Changes from Exp2/Exp3:
  - p1-p99 normalized PNG dataset with compressed COCO RLE masks.
  - Exp2 train augmentation: hflip + vflip + photometric distortion.
  - Exp3c loss weighting: bbox loss x1.25 and mask loss x1.25.
  - Classmate-style inference head settings: score_thr=0.001, max_per_img=300.
"""

_base_ = "./cascade_mask_rcnn_r50_fpn_hw3_adamw_cosine.py"

train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(type="Resize", scale=(1024, 1024), keep_ratio=True),
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

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        ann_file="annotations_norm_png_rle/instances_hw3_train.json",
        data_prefix=dict(img="data_norm_png_rle/train/"),
        pipeline=train_pipeline,
    )
)

val_dataloader = dict(
    dataset=dict(
        ann_file="annotations_norm_png_rle/instances_hw3_val.json",
        data_prefix=dict(img="data_norm_png_rle/val/"),
    )
)

test_dataloader = dict(
    dataset=dict(
        ann_file="annotations_norm_png_rle/image_info_hw3_test.json",
        data_prefix=dict(img="data_norm_png_rle/test_release/"),
    )
)

val_evaluator = dict(ann_file="./annotations_norm_png_rle/instances_hw3_val.json")
test_evaluator = dict(
    ann_file="./annotations_norm_png_rle/image_info_hw3_test.json",
    outfile_prefix="submissions/hw3_exp4_test",
)

model = dict(
    roi_head=dict(
        bbox_head=[
            dict(
                type="Shared2FCBBoxHead",
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=4,
                bbox_coder=dict(
                    type="DeltaXYWHBBoxCoder",
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2],
                ),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0
                ),
                loss_bbox=dict(type="SmoothL1Loss", beta=1.0, loss_weight=1.25),
            ),
            dict(
                type="Shared2FCBBoxHead",
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=4,
                bbox_coder=dict(
                    type="DeltaXYWHBBoxCoder",
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.05, 0.05, 0.1, 0.1],
                ),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0
                ),
                loss_bbox=dict(type="SmoothL1Loss", beta=1.0, loss_weight=1.25),
            ),
            dict(
                type="Shared2FCBBoxHead",
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=4,
                bbox_coder=dict(
                    type="DeltaXYWHBBoxCoder",
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.033, 0.033, 0.067, 0.067],
                ),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0
                ),
                loss_bbox=dict(type="SmoothL1Loss", beta=1.0, loss_weight=1.25),
            ),
        ],
        mask_head=dict(
            type="FCNMaskHead",
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=4,
            loss_mask=dict(type="CrossEntropyLoss", use_mask=True, loss_weight=1.25),
        ),
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type="nms", iou_threshold=0.7),
            min_bbox_size=0,
        ),
        rcnn=dict(
            score_thr=0.001,
            nms=dict(type="nms", iou_threshold=0.5),
            max_per_img=300,
            mask_thr_binary=0.5,
        ),
    ),
)

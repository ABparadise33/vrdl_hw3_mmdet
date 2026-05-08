"""Cascade Mask R-CNN R50-FPN config for VRDL HW3.

This config targets MMDetection 3.x. It inherits the official COCO Cascade Mask
R-CNN config and overrides dataset paths, class counts, training schedule, and
test-time augmentation.
"""

_base_ = "mmdet::cascade_rcnn/cascade-mask-rcnn_r50_fpn_1x_coco.py"

classes = ("class1", "class2", "class3", "class4")
data_root = "./"

model = dict(
    roi_head=dict(
        bbox_head=[
            dict(num_classes=4),
            dict(num_classes=4),
            dict(num_classes=4),
        ],
        mask_head=dict(num_classes=4),
    )
)

load_from = (
    "https://download.openmmlab.com/mmdetection/v2.0/"
    "cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco/"
    "cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth"
)

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        ann_file="annotations/instances_hw3_train.json",
        data_prefix=dict(img="data/train/"),
        metainfo=dict(classes=classes),
        filter_cfg=dict(filter_empty_gt=True, min_size=1),
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        ann_file="annotations/instances_hw3_val.json",
        data_prefix=dict(img="data/train/"),
        metainfo=dict(classes=classes),
        test_mode=True,
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        ann_file="annotations/image_info_hw3_test.json",
        data_prefix=dict(img="data/test_release/"),
        metainfo=dict(classes=classes),
        test_mode=True,
    ),
)

val_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + "annotations/instances_hw3_val.json",
    metric=["bbox", "segm"],
    classwise=True,
)

test_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + "annotations/image_info_hw3_test.json",
    metric=["bbox", "segm"],
    format_only=True,
    outfile_prefix="submissions/hw3_test",
)

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="SGD", lr=0.0025, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2),
)

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=24, val_interval=1)
param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type="MultiStepLR",
        begin=0,
        end=24,
        by_epoch=True,
        milestones=[16, 22],
        gamma=0.1,
    ),
]

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        interval=1,
        save_best="coco/segm_mAP_50",
        rule="greater",
        max_keep_ckpts=5,
    ),
    logger=dict(type="LoggerHook", interval=20),
)

tta_model = dict(
    type="DetTTAModel",
    tta_cfg=dict(nms=dict(type="nms", iou_threshold=0.5), max_per_img=300),
)

tta_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(
        type="TestTimeAug",
        transforms=[
            [
                dict(type="Resize", scale=(800, 800), keep_ratio=True),
                dict(type="Resize", scale=(1000, 1000), keep_ratio=True),
                dict(type="Resize", scale=(1200, 1200), keep_ratio=True),
            ],
            [
                dict(type="RandomFlip", prob=1.0, direction="horizontal"),
                dict(type="RandomFlip", prob=0.0),
            ],
            [
                dict(
                    type="PackDetInputs",
                    meta_keys=(
                        "img_id",
                        "img_path",
                        "ori_shape",
                        "img_shape",
                        "scale_factor",
                        "flip",
                        "flip_direction",
                    ),
                )
            ],
        ],
    ),
]

vis_backends = [
    dict(type="LocalVisBackend"),
]
visualizer = dict(
    type="DetLocalVisualizer",
    vis_backends=vis_backends,
    name="visualizer",
)

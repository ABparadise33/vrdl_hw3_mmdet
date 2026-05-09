"""Exp1: AdamW/cosine baseline on p1-p99 normalized 3-channel PNG images."""

_base_ = "./cascade_mask_rcnn_r50_fpn_hw3_adamw_cosine.py"

train_dataloader = dict(
    dataset=dict(
        ann_file="annotations_norm_png/instances_hw3_train.json",
        data_prefix=dict(img="data_norm_png/train/"),
    )
)

val_dataloader = dict(
    dataset=dict(
        ann_file="annotations_norm_png/instances_hw3_val.json",
        data_prefix=dict(img="data_norm_png/val/"),
    )
)

test_dataloader = dict(
    dataset=dict(
        ann_file="annotations_norm_png/image_info_hw3_test.json",
        data_prefix=dict(img="data_norm_png/test_release/"),
    )
)

val_evaluator = dict(ann_file="./annotations_norm_png/instances_hw3_val.json")
test_evaluator = dict(
    ann_file="./annotations_norm_png/image_info_hw3_test.json",
    outfile_prefix="submissions/hw3_norm_png_test",
)

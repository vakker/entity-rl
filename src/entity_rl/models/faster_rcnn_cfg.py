# Faster R-CNN Configuration for ENROS Entity Encoder

_base_ = "faster_rcnn_base_cfg.py"

# Use pre-trained Faster R-CNN checkpoint from mmdetection model zoo
# chkp = "faster_rcnn_r50_fpn_2x_coco_.......pth"
chkp = "mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth"

model = dict(
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type="nms", iou_threshold=0.7),
            min_bbox_size=0,
        ),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type="nms", iou_threshold=0.5),
            max_per_img=100,
        ),
    ),
    # Keep backbone mostly frozen initially
    backbone=dict(
        frozen_stages=-1  # Freeze first 2 ResNet stages
    ),
)

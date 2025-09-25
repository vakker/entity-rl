# RPN Configuration for ENROS Entity Encoder

_base_ = "rpn_base_cfg.py"

chkp = "mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth"

model = dict(
    # Reduce NMS threshold for more proposals in RL setting
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=50,  # Reasonable number for RL
            nms=dict(type='nms', iou_threshold=0.5),  # More permissive
            min_bbox_size=0
        )
    ),
    # Keep backbone mostly frozen initially
    backbone=dict(
        frozen_stages=-1  # Freeze first 2 ResNet stages
    ),
)

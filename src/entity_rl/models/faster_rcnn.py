# pylint: skip-file
# flake8: noqa

import copy

from functools import partial
from mmdet.utils import ConfigType, InstanceList
from typing import Dict, Optional, Tuple, Union, List

import torch
import torch.nn as nn
from mmdet.models.detectors.faster_rcnn import FasterRCNN
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample, OptSampleList, SampleList
from mmdet.structures.bbox import bbox2roi
from mmdet.utils import ConfigType
from mmengine.registry import DefaultScope
from torch import Tensor

from .base import freeze, unfreeze

# This is needed for MM to handle the registry properly
_ = DefaultScope.get_instance("EXPERIMENT", scope_name="mmdet")


def patch_predict_bbox(
    self,
    x: Tuple[Tensor],
    batch_img_metas: List[dict],
    rpn_results_list: InstanceList,
    rcnn_test_cfg: ConfigType,
    rescale: bool = False,
) -> InstanceList:
    proposals = [res.bboxes for res in rpn_results_list]
    rois = bbox2roi(proposals)

    if rois.shape[0] == 0:
        raise NotImplementedError
        # return empty_instances(
        #     batch_img_metas,
        #     rois.device,
        #     task_type='bbox',
        #     box_type=self.bbox_head.predict_box_type,
        #     num_classes=self.bbox_head.num_classes,
        #     score_per_cls=rcnn_test_cfg is None)

    bbox_results = self._bbox_forward(x, rois)
    self.bbox_feats = bbox_results["bbox_feats"]

    # split batch bbox prediction back to each image
    cls_scores = bbox_results["cls_score"]
    bbox_preds = bbox_results["bbox_pred"]
    num_proposals_per_img = tuple(len(p) for p in proposals)
    rois = rois.split(num_proposals_per_img, 0)
    cls_scores = cls_scores.split(num_proposals_per_img, 0)

    # some detector with_reg is False, bbox_preds will be None
    if bbox_preds is not None:
        # TODO move this to a sabl_roi_head
        # the bbox prediction of some detectors like SABL is not Tensor
        if isinstance(bbox_preds, torch.Tensor):
            bbox_preds = bbox_preds.split(num_proposals_per_img, 0)
        else:
            bbox_preds = self.bbox_head.bbox_pred_split(
                bbox_preds, num_proposals_per_img
            )
    else:
        bbox_preds = (None,) * len(proposals)

    self.bbox_preds = bbox_preds

    result_list = self.bbox_head.predict_by_feat(
        rois=rois,
        cls_scores=cls_scores,
        bbox_preds=bbox_preds,
        batch_img_metas=batch_img_metas,
        rcnn_test_cfg=rcnn_test_cfg,
        rescale=rescale,
    )
    return result_list


# @MODELS.register_module()
class FasterRCNNENROS(FasterRCNN):
    """Faster R-CNN detector customized for ENROS entity extraction.

    Unlike RPNENROS which only uses RPN proposals, this class uses the full
    two-stage pipeline (RPN + RoI head) to get refined bounding boxes with
    class predictions before extracting features.
    """

    def __init__(
        self,
        roi_output_size: int = 7,
        unfreeze_backbone: bool = False,
        unfreeze_roi_head: bool = False,
        *args,
        **kwargs,
    ) -> None:
        self.roi_output_size = roi_output_size
        self.unfreeze_backbone = unfreeze_backbone
        self.unfreeze_roi_head = unfreeze_roi_head

        # Calculate actual feature dimension: 256 channels * roi_size * roi_size
        self.feature_dim = 256 * roi_output_size * roi_output_size

        super().__init__(*args, **kwargs)

        self.roi_head.predict_bbox = partial(patch_predict_bbox, self.roi_head)

        # Freeze everything by default
        freeze(self)

        # Selectively unfreeze based on config
        if unfreeze_backbone:
            unfreeze(self.backbone)

        if unfreeze_roi_head:
            unfreeze(self.roi_head)

    def forward(
        self,
        batch_inputs: Tensor,
        batch_data_samples: OptSampleList = None,
        mode: str = "tensor",
    ):
        """Forward pass that returns refined detections and RoI features.

        Args:
            batch_inputs: Input images of shape (N, C, H, W)
            batch_data_samples: Data samples (for compatibility)
            mode: Forward mode - 'tensor' returns features dict

        Returns:
            dict: Dictionary containing:
                - detections: List of DetDataSample with refined bbox detections
                - roi_features: Tensor of shape (N, max_detections, feature_dim)
                - backbone_features: Backbone feature maps
        """
        if mode == "tensor":
            return self._forward_tensor(batch_inputs, batch_data_samples)
        else:
            # Standard Faster R-CNN forward for training/prediction
            return super().forward(batch_inputs, batch_data_samples, mode)

    def _forward_tensor(
        self, batch_inputs: Tensor, batch_data_samples: OptSampleList = None
    ):
        """Forward pass returning features for ENROS."""

        # Extract backbone features
        x = self.extract_feat(batch_inputs)

        # Generate RPN proposals
        if batch_data_samples is None:
            batch_data_samples = []
            for b in range(batch_inputs.shape[0]):
                data_sample = DetDataSample()
                data_sample.set_metainfo(
                    dict(
                        img_shape=batch_inputs.shape[2:],
                        batch_input_shape=batch_inputs.shape[2:],
                        scale_factor=(1.0, 1.0),
                    )
                )
                batch_data_samples.append(data_sample)

        # Get proposals from RPN head
        rpn_results_list = self.rpn_head.predict(x, batch_data_samples, rescale=False)

        # Get refined detections from RoI head
        # This performs bbox regression and classification
        detections_list = self.roi_head.predict_bbox(
            x=x,
            batch_img_metas=[ds.metainfo for ds in batch_data_samples],
            rpn_results_list=rpn_results_list,
            rcnn_test_cfg=self.test_cfg.rcnn,
            rescale=False,
        )

        # Extract RoI features for each detection
        batch_size = len(detections_list)
        max_detections = max(
            len(dets.bboxes) if len(dets.bboxes) > 0 else 1 for dets in detections_list
        )

        # Prepare output tensors
        all_features = torch.zeros(
            batch_size,
            max_detections,
            self.feature_dim,
            device=batch_inputs.device,
            dtype=batch_inputs.dtype,
        )

        all_preds = torch.zeros(
            batch_size,
            max_detections,
            320,
            device=batch_inputs.device,
            dtype=batch_inputs.dtype,
        )

        all_bboxes = torch.zeros(
            batch_size,
            max_detections,
            4,
            device=batch_inputs.device,
            dtype=batch_inputs.dtype,
        )

        all_scores = torch.zeros(
            batch_size,
            max_detections,
            device=batch_inputs.device,
            dtype=batch_inputs.dtype,
        )

        all_labels = torch.zeros(
            batch_size,
            max_detections,
            device=batch_inputs.device,
            dtype=torch.long,
        )

        # Process each batch item
        for batch_idx, detections in enumerate(detections_list):
            if len(detections.bboxes) == 0:
                # No detections - create dummy features
                continue

            # Get bounding boxes, scores, and labels
            bboxes = detections.bboxes  # (N, 4) - refined bboxes
            scores = detections.scores  # (N,) - class scores
            labels = detections.labels  # (N,) - class labels

            # Add batch index to bboxes: [batch_idx, x1, y1, x2, y2]
            batch_indices = torch.full(
                (bboxes.shape[0], 1),
                batch_idx,
                device=bboxes.device,
                dtype=bboxes.dtype,
            )
            rois = torch.cat([batch_indices, bboxes], dim=1)

            # Extract RoI features using P2 feature map (stride=4)
            pooled_features = self.roi_head.bbox_feats
            pooled_features = pooled_features.flatten(1)  # (N, 256 * 7 * 7)

            assert len(self.roi_head.bbox_preds) == 1
            bbox_preds = self.roi_head.bbox_preds[0]
            __import__('ipdb').set_trace()

            # Store features and metadata
            n_detections = min(bboxes.shape[0], max_detections)
            all_features[batch_idx, :n_detections] = pooled_features[:n_detections]
            all_bboxes[batch_idx, :n_detections] = bboxes[:n_detections]
            all_scores[batch_idx, :n_detections] = scores[:n_detections]
            all_labels[batch_idx, :n_detections] = labels[:n_detections]
            all_preds[batch_idx, :n_detections] = bbox_preds[:n_detections]

            # print(all_scores)

        return {
            "detections": detections_list,
            "features": all_features,
            "preds": all_preds,
            "bboxes": all_bboxes,
            "scores": all_scores,
            "labels": all_labels,
            "backbone_features": x,
        }

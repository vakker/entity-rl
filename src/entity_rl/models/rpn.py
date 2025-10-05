# pylint: skip-file
# flake8: noqa

import copy
from typing import Dict, Optional, Tuple, Union, List

import torch
import torch.nn as nn
from mmdet.models.detectors.rpn import RPN
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample, OptSampleList, SampleList
from mmdet.utils import ConfigType
from mmengine.registry import DefaultScope
from torch import Tensor

from .base import freeze, unfreeze

# This is needed for MM to handle the registry properly
_ = DefaultScope.get_instance("EXPERIMENT", scope_name="mmdet")


class RPNENROS(RPN):
    """RPN detector customized for ENROS entity extraction."""

    def __init__(
        self,
        roi_output_size: int = 7,
        roi_spatial_scale: float = 0.25,  # 1/4 for P2 feature level
        unfreeze_backbone: bool = False,
        *args,
        **kwargs,
    ) -> None:
        self.roi_output_size = roi_output_size
        self.roi_spatial_scale = roi_spatial_scale
        self.unfreeze_backbone = unfreeze_backbone

        # Calculate actual feature dimension: 256 channels * roi_size * roi_size
        # self.pooled_dim = 256
        self.feature_dim = 256 * roi_output_size * roi_output_size

        super().__init__(*args, **kwargs)

        # Add RoI pooling for feature extraction
        from mmcv.ops import RoIAlign

        self.roi_align = RoIAlign(
            output_size=roi_output_size,
            spatial_scale=roi_spatial_scale,
            sampling_ratio=0,
        )

        freeze(self)

        # Freeze/unfreeze based on config
        if unfreeze_backbone:
            unfreeze(self.backbone)

    def freeze_backbone(self):
        """Freeze backbone parameters."""
        raise NotImplementedError

        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        raise NotImplementedError

        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(
        self,
        batch_inputs: Tensor,
        batch_data_samples: OptSampleList = None,
        mode: str = "tensor",
    ):
        """Forward pass that returns both proposals and RoI features.

        Args:
            batch_inputs: Input images of shape (N, C, H, W)
            batch_data_samples: Data samples (for compatibility)
            mode: Forward mode - 'tensor' returns features dict

        Returns:
            dict: Dictionary containing:
                - proposals: List of DetDataSample with bbox proposals
                - roi_features: Tensor of shape (N, max_proposals, feature_dim)
                - backbone_features: Backbone feature maps
        """
        if mode == "tensor":
            return self._forward_tensor(batch_inputs, batch_data_samples)
        else:
            # Standard RPN forward for training/prediction
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
        proposals_list = self.bbox_head.predict(x, batch_data_samples, rescale=False)

        # Extract RoI features for each proposal
        batch_size = len(proposals_list)
        max_proposals = max(
            len(props.bboxes) if len(props.bboxes) > 0 else 1
            for props in proposals_list
        )

        # Prepare output tensors
        all_features = torch.zeros(
            batch_size,
            max_proposals,
            self.feature_dim,
            device=batch_inputs.device,
            dtype=batch_inputs.dtype,
        )

        all_bboxes = torch.zeros(
            batch_size,
            max_proposals,
            4,
            device=batch_inputs.device,
            dtype=batch_inputs.dtype,
        )

        all_scores = torch.zeros(
            batch_size,
            max_proposals,
            1,
            device=batch_inputs.device,
            dtype=batch_inputs.dtype,
        )

        # Process each batch item
        for batch_idx, proposals in enumerate(proposals_list):
            if len(proposals.bboxes) == 0:
                # No proposals - create dummy features
                continue

            # Get bounding boxes and scores
            bboxes = proposals.bboxes  # (N, 4)
            scores = proposals.scores  # (N,)

            # Add batch index to bboxes: [batch_idx, x1, y1, x2, y2]
            batch_indices = torch.full(
                (bboxes.shape[0], 1),
                batch_idx,
                device=bboxes.device,
                dtype=bboxes.dtype,
            )
            rois = torch.cat([batch_indices, bboxes], dim=1)

            # Extract RoI features using P2 feature map (stride=4)
            pooled_features = self.roi_align(x[0], rois)  # (N, 256, roi_size, roi_size)
            # pooled_features = nn.functional.max_pool2d(
            #     pooled_features,
            #     kernel_size=[pooled_features.shape[2], pooled_features.shape[3]],
            # )
            pooled_features = pooled_features.flatten(1)  # (N, 256*roi_size*roi_size)

            # Store features and metadata
            n_proposals = min(bboxes.shape[0], max_proposals)
            all_features[batch_idx, :n_proposals] = pooled_features[:n_proposals]
            all_bboxes[batch_idx, :n_proposals] = bboxes[:n_proposals]
            all_scores[batch_idx, :n_proposals] = scores[:n_proposals].unsqueeze(1)

        return {
            "proposals": proposals_list,
            "features": all_features,
            "bboxes": all_bboxes,
            "scores": all_scores,
            "backbone_features": x,
        }

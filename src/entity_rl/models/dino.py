# pylint: skip-file
# flake8: noqa

"""
DINO (DETR with Improved DeNoising Anchor Boxes) detector wrapper for ENROS.

This module provides a wrapper around MMDetection's DINO detector to extract
condensed object-agnostic features from transformer query embeddings instead
of traditional RoI pooling features.
"""

import copy
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmdet.models.detectors.dino import DINO
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample, OptSampleList, SampleList
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy
from mmdet.utils import ConfigType
from mmengine.registry import DefaultScope
from mmengine.structures import InstanceData
from torch import Tensor
from torch.nn import functional as F

from .base import freeze, unfreeze

# This is needed for MM to handle the registry properly
_ = DefaultScope.get_instance("EXPERIMENT", scope_name="mmdet")


def patch_predict_by_feat_single(
    bbox_head,
    dino_model,
    cls_score: Tensor,
    bbox_pred: Tensor,
    img_meta: dict,
    rescale: bool = True,
) -> InstanceData:
    """Patched version of _predict_by_feat_single that captures query indices.

    This function wraps the original DINOHead._predict_by_feat_single to capture
    which queries were selected (bbox_index) so we can match query embeddings
    to final detections.
    """
    assert len(cls_score) == len(bbox_pred)  # num_queries
    max_per_img = bbox_head.test_cfg.get('max_per_img', len(cls_score))
    img_shape = img_meta['img_shape']

    # exclude background
    if bbox_head.loss_cls.use_sigmoid:
        cls_score = cls_score.sigmoid()
        scores, indexes = cls_score.view(-1).topk(max_per_img)
        det_labels = indexes % bbox_head.num_classes
        bbox_index = indexes // bbox_head.num_classes  # ← Query indices we need!
        bbox_pred = bbox_pred[bbox_index]
    else:
        scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
        scores, bbox_index = scores.topk(max_per_img)
        bbox_pred = bbox_pred[bbox_index]
        det_labels = det_labels[bbox_index]

    # Store the query indices in the model
    dino_model.query_indices.append(bbox_index.cpu())

    det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
    det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
    det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
    det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
    det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
    if rescale:
        assert img_meta.get('scale_factor') is not None
        det_bboxes /= det_bboxes.new_tensor(
            img_meta['scale_factor']).repeat((1, 2))

    results = InstanceData()
    results.bboxes = det_bboxes
    results.scores = scores
    results.labels = det_labels
    return results


class DINOENROS(DINO):
    """DINO detector customized for ENROS entity extraction.

    Unlike traditional detectors that use RoI pooling, DINO uses transformer
    queries to represent objects. This class extracts the decoder query embeddings
    as condensed, object-agnostic features for downstream GNN processing.

    Key advantages over RoI-based approaches:
    - Condensed features: 256D vs 12,544D (256×7×7) from RoI pooling
    - Object-agnostic: Query-based representations without class biases
    - Pretrained on COCO: Strong detection performance (49.4-63.3 AP)
    """

    def __init__(
        self,
        query_dim: int = 256,
        unfreeze_backbone: bool = False,
        unfreeze_encoder: bool = False,
        unfreeze_decoder: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """Initialize DINOENROS detector.

        Args:
            query_dim: Dimension of query embeddings (default: 256)
            unfreeze_backbone: Whether to train the backbone
            unfreeze_encoder: Whether to train the transformer encoder
            unfreeze_decoder: Whether to train the transformer decoder
        """
        self.query_dim = query_dim
        self.unfreeze_backbone = unfreeze_backbone
        self.unfreeze_encoder = unfreeze_encoder
        self.unfreeze_decoder = unfreeze_decoder

        super().__init__(*args, **kwargs)

        self.hidden_state = None
        self.query_indices = []

        # Patch the bbox_head's predict method to capture query indices
        self.bbox_head._predict_by_feat_single = partial(
            patch_predict_by_feat_single, self.bbox_head, self
        )

        # Freeze entire model by default
        freeze(self)

        # Selectively unfreeze components based on config
        if unfreeze_backbone:
            unfreeze(self.backbone)
        if unfreeze_encoder:
            unfreeze(self.neck)  # Neck includes encoder
        if unfreeze_decoder:
            unfreeze(self.bbox_head)  # Head includes decoder

    def predict(
        self,
        batch_inputs: Tensor,
        batch_data_samples: SampleList,
        rescale: bool = True,
    ) -> SampleList:
        # Reset query indices for this forward pass
        self.query_indices = []

        img_feats = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(img_feats, batch_data_samples)
        self.hidden_state = head_inputs_dict['hidden_states'][-1]

        results_list = self.bbox_head.predict(
            **head_inputs_dict, rescale=rescale, batch_data_samples=batch_data_samples
        )
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list
        )
        return batch_data_samples

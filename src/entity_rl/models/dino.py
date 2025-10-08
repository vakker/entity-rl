# pylint: skip-file
# flake8: noqa

"""
DINO (DETR with Improved DeNoising Anchor Boxes) detector wrapper for ENROS.

This module provides a wrapper around MMDetection's DINO detector to extract
condensed object-agnostic features from transformer query embeddings instead
of traditional RoI pooling features.
"""

import copy
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmdet.models.detectors.dino import DINO
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample, OptSampleList, SampleList
from mmdet.utils import ConfigType
from mmengine.registry import DefaultScope
from torch import Tensor

from .base import freeze, unfreeze

# This is needed for MM to handle the registry properly
_ = DefaultScope.get_instance("EXPERIMENT", scope_name="mmdet")


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

        self.hidden_state=None

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

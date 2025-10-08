import copy
import sys
from abc import abstractmethod
from os import path as osp

import torch
from gymnasium import spaces
from mmengine import Config
from mmengine.runner.checkpoint import _load_checkpoint, _load_checkpoint_to_model
from torch_geometric.data import Batch, Data

from entity_rl.models import faster_rcnn

from .base import BaseModule
from .dino import DINOENROS
from .gdino import GDino
from .rpn import RPNENROS

module = sys.modules[__name__]


class EntityEncoder(BaseModule):
    def __init__(self, model_config, obs_space):
        super().__init__()

        self._config = model_config
        self._obs_space = obs_space

    @abstractmethod
    def forward(self, inputs):
        pass

    @property
    @abstractmethod
    def out_channels(self) -> dict:
        pass


class EntityPassThrough(EntityEncoder):
    def __init__(self, model_config, obs_space):
        # NOTE: RLlib currently supports dict and repeated spaces
        assert isinstance(obs_space, spaces.Dict)
        # assert isinstance(obs_space, spaces.Graph)
        super().__init__(model_config, obs_space)

        self._edge_index_map = {}

    @property
    def out_channels(self):
        out_channels = {
            "node_features": self._obs_space["x"].child_space.shape,
            # "edge_features": self._obs_space.edge_features.shape,
            "edge_features": None,
            "global_features": None,
        }
        return out_channels

    def forward(self, inputs):
        batch = Batch(**inputs)
        return batch

        if isinstance(inputs, Batch):
            return inputs

        g_batch = []

        batch_size = inputs["x"].values.shape[0]
        device = inputs["x"].values.device

        if "edge_index" in inputs:
            edge_index = torch.transpose(inputs["edge_index"].values, 2, 1).long()
            edge_index_len = inputs["edge_index"].lengths.long()

        else:
            edge_index = (None for _ in range(batch_size))
            edge_index_len = (None for _ in range(batch_size))

        data = zip(
            inputs["x"].values,
            inputs["x"].lengths.long(),
            edge_index,
            edge_index_len,
        )

        # TODO: the stacking is still a bit slow
        for x, x_len, edge_index, ei_len in data:
            if not x_len:
                x_len = 1
                ei_len = 1

            if edge_index is None:
                n_nodes = x_len
                if not isinstance(n_nodes, int):
                    n_nodes = n_nodes.item()

                # print("#", n_nodes, self._edge_index_map.keys())

                if n_nodes in self._edge_index_map:
                    edge_index = self._edge_index_map[n_nodes]

                else:
                    # For refecence:
                    # start_time = time.time()
                    # edge_index = [
                    #     torch.tensor([i, j], device=input_dict["obs_flat"].device)
                    #     for i in range(n_nodes)
                    #     for j in range(n_nodes)
                    # ]
                    # edge_index = torch.transpose(torch.stack(edge_index), 1, 0).long()
                    # print("edge_index", time.time() - start_time)

                    node_indices = torch.tensor(
                        range(n_nodes),
                        dtype=torch.long,
                        device=device,
                    )

                    j_idx = node_indices.tile((n_nodes,))
                    i_idx = node_indices.repeat_interleave(n_nodes)
                    edge_index = torch.stack([i_idx, j_idx], dim=0)

                    self._edge_index_map[n_nodes] = edge_index

                    ei_len = edge_index.shape[1]

            g_batch.append(Data(x=x[:x_len], edge_index=edge_index[:, :ei_len]))

        batch = Batch.from_data_list(g_batch)
        return batch


class GDINOEncoder(EntityEncoder):
    def __init__(self, model_config, obs_space):
        assert isinstance(obs_space, spaces.Box)
        super().__init__(model_config, obs_space)

        self._model_config = model_config
        current_dir = osp.dirname(osp.abspath(__file__))
        gdino_cfg_file = osp.join(current_dir, model_config["gdino_cfg"])
        assert osp.exists(gdino_cfg_file)

        gdino_config_full = Config.fromfile(gdino_cfg_file)
        gdino_chkp_file = osp.join(current_dir, gdino_config_full.chkp)
        gdino_config = gdino_config_full.model
        if "num_queries" in model_config:
            gdino_config["num_queries"] = model_config["num_queries"]

        gdino_config.pop("language_model")
        gdino_config.pop("type")
        self._model = GDino(
            prompt_size=model_config["prompt_size"],
            max_per_image=model_config["max_per_image"],
            unfreeze_backbone=model_config.get("unfreeze_backbone", False),
            **copy.deepcopy(gdino_config),
        )

        chkp = _load_checkpoint(gdino_chkp_file)["state_dict"]
        self.gdino_outputs = None

        # ? "language_model.language_backbone.body.model.embeddings.position_ids"

        to_remove = ["dn_query_generator.", "language_model."]
        for k in list(chkp.keys()):
            for prefix in to_remove:
                if k.startswith(prefix):
                    del chkp[k]

        _load_checkpoint_to_model(self._model, chkp)

        mean = torch.tensor(
            [123.675, 116.28, 103.53],
            dtype=torch.float32,
            device=self.device,
        )
        std = torch.tensor(
            [58.395, 57.12, 57.375],
            dtype=torch.float32,
            device=self.device,
        )
        self.mean = mean.reshape([1, 3, 1, 1])
        self.std = std.reshape([1, 3, 1, 1])

        if self._config.get("add_edges", False):
            stack_depth = obs_space.shape[2] // 3
            n_nodes = model_config["max_per_image"] * stack_depth
            edge_index = [[i, j] for i in range(n_nodes) for j in range(n_nodes)]
            self.edge_index = (
                torch.tensor(edge_index, dtype=torch.long, device=self.device)
                .t()
                .contiguous()
            )

        else:
            self.edge_index = None

    @property
    def out_channels(self):
        # cls_feat, bbox normalized coords  (cx, cy, w, h), stack depth
        # x_shape = 4 + 1
        x_shape = self._model.cls_features + 4 + 1
        return {
            "node_features": [x_shape],
            "edge_features": None,
            "global_features": None,
        }

    def normalize(self, inputs):
        stack_depth = inputs.shape[1] // 3
        inputs = inputs.to(torch.float32)

        mean = self.mean.to(inputs.device)
        std = self.mean.to(inputs.device)

        mean = mean.repeat(1, stack_depth, 1, 1)
        std = std.repeat(1, stack_depth, 1, 1)
        return (inputs - mean) / std

    def forward(self, inputs):
        inputs = inputs.permute(0, 3, 1, 2)
        assert inputs.shape[1] % 3 == 0

        inputs = self.normalize(inputs)
        stack_depth = inputs.shape[1] // 3
        batch_size = inputs.shape[0]

        if self._config.get("parallel-gdino"):
            inputs = inputs.reshape(
                batch_size * stack_depth, 3, inputs.shape[2], inputs.shape[3]
            )
            outputs = self._model.forward(inputs, mode="tensor")

        else:
            # This could be processed in parallel
            # but that would require more memory
            outputs = []
            for i in range(stack_depth):
                frame = inputs[:, 3 * i : 3 * (i + 1)]
                _outputs = self._model.forward(frame, mode="tensor")
                outputs.append(_outputs)

            outputs = {k: torch.cat([o[k] for o in outputs], dim=0) for k in outputs[0]}

        self.gdino_outputs = {}
        for k in outputs:
            outputs[k] = outputs[k].reshape(
                batch_size, stack_depth, *outputs[k].shape[1:]
            )
            self.gdino_outputs[k] = outputs[k].detach().cpu()  # .numpy()

        # outputs_all is shape (batch_size, stack_depth, n_nodes, node_features)

        stack_features = torch.tensor(
            [[[[i]] for i in range(stack_depth)]], device=self.device
        )
        stack_features = stack_features.expand(
            batch_size, stack_depth, outputs["features"].shape[2], -1
        )
        node_features = torch.cat(
            [
                outputs["features"],
                outputs["bboxes"],
                stack_features,
            ],
            dim=3,
        )

        # Frame nodes has shape (batch_size, stack_depth, n_nodes, node_features)
        # The actual number of nodes should be stack_depth x n_nodes
        node_features = node_features.reshape(
            batch_size,
            stack_depth * node_features.shape[2],
            -1,
        )

        g_batch = []
        if self.edge_index is not None and self.edge_index.device != self.device:
            self.edge_index = self.edge_index.to(self.device)

        # This iterates over the batch dimension
        for sample in node_features:
            # To make sure that it's fully connected
            assert self.edge_index.shape[1] == sample.shape[0] ** 2
            g_batch.append(Data(x=sample, edge_index=self.edge_index))

        batch = Batch.from_data_list(g_batch)
        return batch


class RPNEncoder(EntityEncoder):
    """Entity encoder using RPN for object proposal generation."""

    def __init__(self, model_config, obs_space):
        assert isinstance(obs_space, spaces.Box)
        super().__init__(model_config, obs_space)

        self._model_config = model_config

        current_dir = osp.dirname(osp.abspath(__file__))
        rpn_cfg_file = osp.join(current_dir, model_config["rpn_cfg"])
        assert osp.exists(rpn_cfg_file)

        rpn_config_full = Config.fromfile(rpn_cfg_file)
        if hasattr(rpn_config_full, 'chkp') and rpn_config_full.chkp:
            rpn_chkp_file = osp.join(current_dir, rpn_config_full.chkp)
        else:
            rpn_chkp_file = None

        rpn_config = rpn_config_full.model

        # Override config parameters
        if "max_per_image" in model_config:
            rpn_config["test_cfg"]["rpn"]["max_per_img"] = model_config["max_per_image"]
        if "unfreeze_backbone" in model_config:
            rpn_config["unfreeze_backbone"] = model_config["unfreeze_backbone"]

        rpn_config["type"] = "RPNENROS"
        self._model = RPNENROS(**copy.deepcopy(rpn_config))

        # Load checkpoint if available
        if rpn_chkp_file and osp.exists(rpn_chkp_file):
            chkp = _load_checkpoint(rpn_chkp_file)["state_dict"]
            self.rpn_outputs = None

            # Remove incompatible keys for RPN
            to_remove = ["roi_head.", "mask_head."]  # Remove ROI head parts for pure RPN
            for k in list(chkp.keys()):
                for prefix in to_remove:
                    if k.startswith(prefix):
                        del chkp[k]

            _load_checkpoint_to_model(self._model, chkp)

        # Set image normalization parameters
        mean = torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32)
        std = torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32)
        self.register_buffer("mean", mean.reshape([1, 3, 1, 1]))
        self.register_buffer("std", std.reshape([1, 3, 1, 1]))

        # Add edges between all detected objects if specified
        if model_config.get("add_edges", False):
            stack_depth = obs_space.shape[2] // 3
            max_objects = model_config.get("max_per_image", 50)
            n_nodes = max_objects * stack_depth
            edge_index = [[i, j] for i in range(n_nodes) for j in range(n_nodes)]
            self.register_buffer(
                "edge_index",
                torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
            )
        else:
            self.edge_index = None

        self.rpn_outputs = None

    @property
    def out_channels(self):
        x_shape = 256 * 7 * 7
        return {
            "node_features": (x_shape,),
            "edge_features": None,
            "global_features": None,
        }

    def normalize(self, inputs):
        """Normalize inputs to match ImageNet preprocessing."""
        stack_depth = inputs.shape[1] // 3
        inputs = inputs.to(torch.float32)
        assert stack_depth == 1

        mean = self.mean  # .expand(1, stack_depth, -1, -1).reshape(1, -1, 1, 1)
        std = self.std  # .expand(1, stack_depth, -1, -1).reshape(1, -1, 1, 1)

        inputs = inputs[:, [2, 1, 0], :, :]
        return (inputs - mean) / std

    def forward(self, inputs):
        """
        Args:
            inputs: (batch_size, height, width, channels*stack_depth)

        Returns:
            torch_geometric.data.Batch: Graph batch with detected objects as nodes
        """
        inputs = inputs.permute(0, 3, 1, 2)  # (B, C*stack, H, W)
        assert inputs.shape[1] % 3 == 0

        inputs = self.normalize(inputs)
        stack_depth = inputs.shape[1] // 3
        batch_size = inputs.shape[0]

        all_node_features = []
        all_proposals = []

        # Process each frame in the stack
        for stack_idx in range(stack_depth):
            frame = inputs[:, 3 * stack_idx : 3 * (stack_idx + 1)]  # (B, 3, H, W)

            # Extract features using RPN
            # FIXME: not used:
            # with torch.cuda.amp.autocast(enabled=False):
            outputs = self._model.forward(frame, mode="tensor")

            features = outputs["features"]  # (B, max_proposals, feature_dim)
            bboxes = outputs["bboxes"]  # (B, max_proposals, 4)
            scores = outputs["scores"]  # (B, max_proposals, 1)
            proposals = outputs["proposals"]

            # Normalize bounding boxes to [0, 1]
            h, w = frame.shape[2:]
            norm_bboxes = bboxes.clone()
            norm_bboxes[:, :, [0, 2]] /= w
            norm_bboxes[:, :, [1, 3]] /= h

            # Add frame indices
            frame_ids = torch.full(
                (batch_size, features.shape[1], 1),
                stack_idx,
                device=inputs.device,
                dtype=torch.float32,
            )

            # Combine all features: [roi_features, norm_bboxes, scores, frame_ids]
            node_features = features
            # node_features = torch.cat(
            #     [
            #         features,  # (B, N, feature_dim)
            #         norm_bboxes,  # (B, N, 4)
            #         scores,  # (B, N, 1)
            #         frame_ids,  # (B, N, 1)
            #     ],
            #     dim=2,
            # )  # (B, N, feature_dim + 6)

            all_node_features.append(node_features)
            all_proposals.extend(proposals)

        # Store for visualization/debugging
        self.rpn_outputs = {
            "proposals": all_proposals,
            "features": [f.detach().cpu() for f in all_node_features],
        }

        # Create graph batch
        g_batch = []
        for batch_idx in range(batch_size):
            # Collect all nodes for this batch item across all frames
            batch_nodes = []
            for stack_features in all_node_features:
                batch_nodes.append(stack_features[batch_idx])  # (N, features)

            if batch_nodes:
                all_nodes = torch.cat(batch_nodes, dim=0)  # (total_objects, features)
            else:
                # Fallback: single dummy node
                all_nodes = torch.zeros(
                    1, self.out_channels["node_features"][0], device=inputs.device
                )

            # Remove zero-padded proposals (where all features are zero)
            non_zero_mask = all_nodes.abs().sum(dim=1) > 1e-6
            if non_zero_mask.any():
                all_nodes = all_nodes[non_zero_mask]
            else:
                # Keep at least one node
                all_nodes = all_nodes[:1]

            # Create edges
            n_nodes = all_nodes.shape[0]
            if self.edge_index is not None and self.edge_index.shape[1] >= n_nodes**2:
                # Use precomputed fully connected edges
                edge_index = (
                    self.edge_index[:, : n_nodes**2]
                    .view(2, n_nodes, n_nodes)[:, :n_nodes, :n_nodes]
                    .view(2, -1)
                )
            else:
                # Create fully connected graph
                sources = torch.arange(n_nodes, device=inputs.device).repeat(n_nodes)
                targets = torch.arange(n_nodes, device=inputs.device).repeat_interleave(
                    n_nodes
                )
                edge_index = torch.stack([sources, targets], dim=0)

            g_batch.append(Data(x=all_nodes, edge_index=edge_index))

        batch = Batch.from_data_list(g_batch)
        return batch


class FasterRCNNEncoder(EntityEncoder):
    """Entity encoder using Faster R-CNN for refined object detection."""

    def __init__(self, model_config, obs_space):
        assert isinstance(obs_space, spaces.Box)
        super().__init__(model_config, obs_space)

        self._model_config = model_config

        # Import FasterRCNNENROS
        from .faster_rcnn import FasterRCNNENROS

        current_dir = osp.dirname(osp.abspath(__file__))
        frcnn_cfg_file = osp.join(current_dir, model_config["frcnn_cfg"])
        assert osp.exists(frcnn_cfg_file)

        from mmengine import Config

        frcnn_config_full = Config.fromfile(frcnn_cfg_file)
        if hasattr(frcnn_config_full, 'chkp') and frcnn_config_full.chkp:
            frcnn_chkp_file = osp.join(current_dir, frcnn_config_full.chkp)
        else:
            frcnn_chkp_file = None

        frcnn_config = frcnn_config_full.model

        # Override config parameters
        if "max_per_image" in model_config:
            frcnn_config["test_cfg"]["rcnn"]["max_per_img"] = model_config["max_per_image"]
        if "unfreeze_backbone" in model_config:
            frcnn_config["unfreeze_backbone"] = model_config["unfreeze_backbone"]
        if "unfreeze_roi_head" in model_config:
            frcnn_config["unfreeze_roi_head"] = model_config["unfreeze_roi_head"]

        # frcnn_config["type"] = "FasterRCNNENROS"
        frcnn_config.pop("type")
        self._model = FasterRCNNENROS(**copy.deepcopy(frcnn_config))

        # Load checkpoint if available
        if frcnn_chkp_file and osp.exists(frcnn_chkp_file):
            chkp = _load_checkpoint(frcnn_chkp_file)["state_dict"]
            self.frcnn_outputs = None

            # Remove incompatible keys for RPN
            to_remove = ["roi_head.mask_head."]  # Remove ROI head parts for pure RPN
            for k in list(chkp.keys()):
                for prefix in to_remove:
                    if k.startswith(prefix):
                        del chkp[k]

            # Load full checkpoint (backbone, RPN, and RoI head)
            _load_checkpoint_to_model(self._model, chkp)

        else:
            raise ValueError("Faster R-CNN checkpoint not found.")

        # Set image normalization parameters
        mean = torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32)
        std = torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32)
        self.register_buffer("mean", mean.reshape([1, 3, 1, 1]))
        self.register_buffer("std", std.reshape([1, 3, 1, 1]))

        # Add edges between all detected objects if specified
        if model_config.get("add_edges", False):
            stack_depth = obs_space.shape[2] // 3
            max_objects = model_config.get("max_per_image", 100)
            n_nodes = max_objects * stack_depth
            edge_index = [[i, j] for i in range(n_nodes) for j in range(n_nodes)]
            self.register_buffer(
                "edge_index",
                torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
            )
        else:
            self.edge_index = None

        self.frcnn_outputs = None

    @property
    def out_channels(self):
        # Feature dimension from RoI pooling: 256 channels
        x_shape = 256
        return {
            "node_features": (x_shape,),
            "edge_features": None,
            "global_features": None,
        }

    def normalize(self, inputs):
        """Normalize inputs to match ImageNet preprocessing.

        Also converts RGB to BGR since the model was trained on BGR images.
        """
        stack_depth = inputs.shape[1] // 3
        inputs = inputs.to(torch.float32)
        assert stack_depth == 1

        # Convert RGB to BGR (flip channels)
        # Input is (B, 3, H, W) in RGB, convert to BGR
        inputs = inputs[:, [2, 1, 0], :, :]

        mean = self.mean
        std = self.std

        return (inputs - mean) / std

    def forward(self, inputs):
        """
        Args:
            inputs: (batch_size, height, width, channels*stack_depth)

        Returns:
            torch_geometric.data.Batch: Graph batch with detected objects as nodes
        """
        inputs = inputs.permute(0, 3, 1, 2)  # (B, C*stack, H, W)
        assert inputs.shape[1] % 3 == 0

        inputs = self.normalize(inputs)
        stack_depth = inputs.shape[1] // 3
        batch_size = inputs.shape[0]

        all_node_features = []
        all_detections = []

        # Process each frame in the stack
        for stack_idx in range(stack_depth):
            frame = inputs[:, 3 * stack_idx : 3 * (stack_idx + 1)]  # (B, 3, H, W)

            # Extract features using Faster R-CNN
            outputs = self._model.forward(frame, mode="tensor")

            features = outputs["features"]  # (B, max_detections, feature_dim)
            bboxes = outputs["bboxes"]  # (B, max_detections, 4)
            scores = outputs["scores"]  # (B, max_detections)
            labels = outputs["labels"]  # (B, max_detections)
            detections = outputs["detections"]

            # Normalize bounding boxes to [0, 1]
            h, w = frame.shape[2:]
            norm_bboxes = bboxes.clone()
            norm_bboxes[:, :, [0, 2]] /= w
            norm_bboxes[:, :, [1, 3]] /= h

            # Add frame indices
            frame_ids = torch.full(
                (batch_size, features.shape[1], 1),
                stack_idx,
                device=inputs.device,
                dtype=torch.float32,
            )

            # Use RoI features directly (256-dim per detection)
            node_features = features

            all_node_features.append(node_features)
            all_detections.extend(detections)

        # Store for visualization/debugging
        self.frcnn_outputs = {
            "detections": all_detections,
            "features": [f.detach().cpu() for f in all_node_features],
            "bboxes": outputs["bboxes"],
            "scores": outputs["scores"],
            "labels": outputs["labels"],
        }

        # Create graph batch
        g_batch = []
        for batch_idx in range(batch_size):
            # Collect all nodes for this batch item across all frames
            batch_nodes = []
            for stack_features in all_node_features:
                batch_nodes.append(stack_features[batch_idx])  # (N, features)

            if batch_nodes:
                all_nodes = torch.cat(batch_nodes, dim=0)  # (total_objects, features)
            else:
                # Fallback: single dummy node
                all_nodes = torch.zeros(
                    1, self.out_channels["node_features"][0], device=inputs.device
                )

            # Remove zero-padded detections (where all features are zero)
            non_zero_mask = all_nodes.abs().sum(dim=1) > 1e-6
            if non_zero_mask.any():
                all_nodes = all_nodes[non_zero_mask]
            else:
                # Keep at least one node
                all_nodes = all_nodes[:1]

            # Create edges
            n_nodes = all_nodes.shape[0]
            if self.edge_index is not None and self.edge_index.shape[1] >= n_nodes**2:
                # Use precomputed fully connected edges
                edge_index = (
                    self.edge_index[:, : n_nodes**2]
                    .view(2, n_nodes, n_nodes)[:, :n_nodes, :n_nodes]
                    .view(2, -1)
                )
            else:
                # Create fully connected graph
                sources = torch.arange(n_nodes, device=inputs.device).repeat(n_nodes)
                targets = torch.arange(n_nodes, device=inputs.device).repeat_interleave(
                    n_nodes
                )
                edge_index = torch.stack([sources, targets], dim=0)

            g_batch.append(Data(x=all_nodes, edge_index=edge_index))

        batch = Batch.from_data_list(g_batch)
        return batch


class DINOEncoder(EntityEncoder):
    """Entity encoder using DINO transformer queries for object detection.

    This encoder uses DINO (DETR with Improved DeNoising Anchor Boxes) to extract
    condensed, object-agnostic features from transformer query embeddings instead
    of traditional RoI pooling features.

    Uses MMDetection's standard predict() method for cleaner integration.

    Key advantages:
    - Condensed features: 256D vs 12,544D from RoI pooling
    - Object-agnostic: Query-based representations without class biases
    - Pretrained on COCO: Strong detection performance (49.4-63.3 AP)
    """

    def __init__(self, model_config, obs_space):
        assert isinstance(obs_space, spaces.Box)
        super().__init__(model_config, obs_space)

        self._model_config = model_config

        # Load DINO config
        from mmengine import Config

        current_dir = osp.dirname(osp.abspath(__file__))
        dino_cfg_file = osp.join(current_dir, model_config["dino_cfg"])
        assert osp.exists(dino_cfg_file), f"DINO config not found: {dino_cfg_file}"

        dino_config_full = Config.fromfile(dino_cfg_file)
        if hasattr(dino_config_full, 'chkp') and dino_config_full.chkp:
            dino_chkp_file = osp.join(current_dir, dino_config_full.chkp)
        else:
            dino_chkp_file = None

        dino_config = dino_config_full.model

        # Override config parameters
        if "max_queries" in model_config:
            # Set maximum number of queries to keep
            if "test_cfg" in dino_config:
                dino_config["test_cfg"]["max_per_img"] = model_config["max_queries"]

        if "score_threshold" in model_config:
            # Set score threshold for filtering detections
            if "test_cfg" in dino_config:
                dino_config["test_cfg"]["score_thr"] = model_config["score_threshold"]

        # Get query dimension and freeze/unfreeze settings
        query_dim = model_config.get("query_dim", 256)
        self._query_dim = query_dim
        self._max_queries = model_config.get("max_queries", 100)
        self._score_threshold = model_config.get("score_threshold", 0.0)

        unfreeze_backbone = model_config.get("unfreeze_backbone", False)
        unfreeze_encoder = model_config.get("unfreeze_encoder", False)
        unfreeze_decoder = model_config.get("unfreeze_decoder", False)

        # Build DINOENROS model with unfreeze settings
        dino_config_dict = dino_config.to_dict()
        # Remove 'type' key as it's not a constructor argument
        dino_config_dict.pop('type', None)
        dino_config_dict['query_dim'] = query_dim
        dino_config_dict['unfreeze_backbone'] = unfreeze_backbone
        dino_config_dict['unfreeze_encoder'] = unfreeze_encoder
        dino_config_dict['unfreeze_decoder'] = unfreeze_decoder

        self._model = DINOENROS(**dino_config_dict)
        self._model.eval()

        # Load checkpoint if available
        if dino_chkp_file and osp.exists(dino_chkp_file):
            chkp = _load_checkpoint(dino_chkp_file)["state_dict"]
            _load_checkpoint_to_model(self._model, chkp)

        # Set image normalization parameters (ImageNet stats)
        mean = torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32)
        std = torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32)
        self.register_buffer("mean", mean.reshape([1, 3, 1, 1]))
        self.register_buffer("std", std.reshape([1, 3, 1, 1]))

        # Add edges between all detected objects if specified
        if model_config.get("add_edges", False):
            stack_depth = obs_space.shape[2] // 3
            max_queries = model_config.get("max_queries", 100)
            n_nodes = max_queries * stack_depth
            edge_index = [[i, j] for i in range(n_nodes) for j in range(n_nodes)]
            self.register_buffer(
                "edge_index",
                torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
            )
        else:
            self.edge_index = None

        self.dino_outputs = None

    @property
    def out_channels(self):
        # Query embedding (256) + bbox (4) + score (1) + frame_id (1) = 262
        node_feat_dim = self._query_dim + 4 + 1 + 1
        return {
            "node_features": (node_feat_dim,),
            "edge_features": None,
            "global_features": None,
        }

    def normalize(self, inputs):
        """Normalize inputs to match ImageNet preprocessing."""
        stack_depth = inputs.shape[1] // 3
        inputs = inputs.to(torch.float32)

        mean = self.mean
        std = self.std

        # Convert BGR to RGB if needed
        inputs = inputs[:, [2, 1, 0], :, :]
        return (inputs - mean) / std

    def forward(self, inputs):
        """
        Args:
            inputs: (batch_size, height, width, channels*stack_depth)

        Returns:
            torch_geometric.data.Batch: Graph batch with DINO queries as nodes
        """
        from mmdet.structures import DetDataSample

        inputs = inputs.permute(0, 3, 1, 2)  # (B, C*stack, H, W)
        assert inputs.shape[1] % 3 == 0

        inputs = self.normalize(inputs)
        stack_depth = inputs.shape[1] // 3
        batch_size = inputs.shape[0]

        all_node_features = []
        all_detections = []

        # Process each frame in the stack
        for stack_idx in range(stack_depth):
            frame = inputs[:, 3 * stack_idx : 3 * (stack_idx + 1)]  # (B, 3, H, W)
            img_h, img_w = frame.shape[2:]

            # Create data samples for MMDetection
            batch_data_samples = []
            for b in range(batch_size):
                data_sample = DetDataSample()
                data_sample.set_metainfo(
                    dict(
                        img_shape=(img_h, img_w),
                        batch_input_shape=(img_h, img_w),
                        scale_factor=(1.0, 1.0),
                    )
                )
                batch_data_samples.append(data_sample)

            # Run DINO predict (stores hidden_state in model)
            with torch.no_grad():
                results_list = self._model.predict(
                    frame, batch_data_samples, rescale=False
                )

            # Extract query features from stored hidden_state
            if self._model.hidden_state is None:
                raise RuntimeError(
                    "Hidden state not captured. Check DINOENROS predict() method."
                )

            query_embeddings = self._model.hidden_state  # (B, num_queries, 256)

            # Process predictions for each batch item
            batch_features_list = []
            batch_bboxes_list = []
            batch_scores_list = []
            batch_labels_list = []

            for batch_idx, result in enumerate(results_list):
                # Get predictions
                pred_bboxes = result.pred_instances.bboxes  # (N, 4) xyxy format
                pred_scores = result.pred_instances.scores  # (N,)
                pred_labels = result.pred_instances.labels  # (N,)

                n_dets = len(pred_bboxes)

                # Filter by score threshold and limit to max_queries
                if n_dets > 0:
                    score_mask = pred_scores > self._score_threshold
                    pred_bboxes = pred_bboxes[score_mask]
                    pred_scores = pred_scores[score_mask]
                    pred_labels = pred_labels[score_mask]
                    n_dets = len(pred_bboxes)

                    # Limit to max_queries
                    if n_dets > self._max_queries:
                        # Keep top-k by score
                        top_k_indices = torch.topk(pred_scores, self._max_queries).indices
                        pred_bboxes = pred_bboxes[top_k_indices]
                        pred_scores = pred_scores[top_k_indices]
                        pred_labels = pred_labels[top_k_indices]
                        n_dets = self._max_queries

                # Get corresponding query features
                # Note: predict() applies NMS and filtering, so we need to match
                # For now, take the first n_dets queries (this is an approximation)
                # A better approach would track query indices through the pipeline
                if n_dets > 0:
                    batch_query_feats = query_embeddings[batch_idx, :n_dets]  # (n_dets, 256)
                else:
                    # No detections, create empty tensors
                    batch_query_feats = torch.zeros(
                        (0, self._query_dim), device=frame.device, dtype=frame.dtype
                    )
                    pred_bboxes = torch.zeros((0, 4), device=frame.device, dtype=frame.dtype)
                    pred_scores = torch.zeros((0,), device=frame.device, dtype=frame.dtype)
                    pred_labels = torch.zeros((0,), device=frame.device, dtype=torch.long)

                batch_features_list.append(batch_query_feats)
                batch_bboxes_list.append(pred_bboxes)
                batch_scores_list.append(pred_scores)
                batch_labels_list.append(pred_labels)

            # Pad to max_queries for batch processing
            max_dets_in_batch = max(len(f) for f in batch_features_list)
            if max_dets_in_batch == 0:
                max_dets_in_batch = 1  # At least 1 to avoid empty tensors

            features = torch.zeros(
                (batch_size, max_dets_in_batch, self._query_dim),
                device=frame.device,
                dtype=frame.dtype,
            )
            bboxes = torch.zeros(
                (batch_size, max_dets_in_batch, 4),
                device=frame.device,
                dtype=frame.dtype,
            )
            scores = torch.zeros(
                (batch_size, max_dets_in_batch, 1),
                device=frame.device,
                dtype=frame.dtype,
            )
            labels = torch.zeros(
                (batch_size, max_dets_in_batch),
                device=frame.device,
                dtype=torch.long,
            )

            for batch_idx in range(batch_size):
                n_dets = len(batch_features_list[batch_idx])
                if n_dets > 0:
                    features[batch_idx, :n_dets] = batch_features_list[batch_idx]
                    bboxes[batch_idx, :n_dets] = batch_bboxes_list[batch_idx]
                    scores[batch_idx, :n_dets, 0] = batch_scores_list[batch_idx]
                    labels[batch_idx, :n_dets] = batch_labels_list[batch_idx]

            # Normalize bboxes to [0, 1]
            bboxes_norm = bboxes.clone()
            bboxes_norm[..., [0, 2]] /= img_w
            bboxes_norm[..., [1, 3]] /= img_h
            bboxes_norm = bboxes_norm.clamp(0, 1)

            # Add frame index
            frame_indices = torch.full(
                (batch_size, features.shape[1], 1),
                stack_idx,
                device=features.device,
                dtype=features.dtype,
            )

            # Concatenate: query_embedding + bbox + score + frame_id
            node_features = torch.cat(
                [features, bboxes_norm, scores, frame_indices], dim=2
            )

            all_node_features.append(node_features)

            # Store detection info for debugging
            detection_info = {
                "features": features,
                "bboxes": bboxes,
                "scores": scores,
                "labels": labels,
            }
            all_detections.append(detection_info)

        # Concatenate all frames
        # Shape: (batch_size, stack_depth * max_queries, node_feat_dim)
        all_node_features = torch.cat(all_node_features, dim=1)

        # Store outputs for visualization/debugging
        self.dino_outputs = all_detections

        # Build graph batch
        g_batch = []

        for batch_idx in range(batch_size):
            all_nodes = all_node_features[batch_idx]  # (total_nodes, node_feat_dim)
            n_nodes = all_nodes.shape[0]

            # Create edges
            if self.edge_index is not None and n_nodes > 0:
                # Use precomputed fully connected edges
                edge_index = (
                    self.edge_index[:, : n_nodes**2]
                    .view(2, n_nodes, n_nodes)[:, :n_nodes, :n_nodes]
                    .view(2, -1)
                )
            else:
                # Create fully connected graph
                sources = torch.arange(n_nodes, device=inputs.device).repeat(n_nodes)
                targets = torch.arange(n_nodes, device=inputs.device).repeat_interleave(
                    n_nodes
                )
                edge_index = torch.stack([sources, targets], dim=0)

            g_batch.append(Data(x=all_nodes, edge_index=edge_index))

        batch = Batch.from_data_list(g_batch)
        return batch

import copy
import sys
from abc import abstractmethod
from os import path as osp

import torch
from gymnasium import spaces
from mmengine import Config
from mmengine.runner.checkpoint import _load_checkpoint, _load_checkpoint_to_model
from torch_geometric.data import Batch, Data

from .base import BaseModule
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
        self.use_precomputed_features = model_config.get("use_precomputed_features", False)
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
        if self.use_precomputed_features:
            # Precomputed features: 261 dims (256 + 5)
            x_shape = 256 + 5
        else:
            # Simple features: 5 dims (rel_x, rel_y, w, h, is_agent)
            x_shape = 5
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

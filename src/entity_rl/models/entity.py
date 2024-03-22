import copy
import sys
from abc import abstractmethod
from os import path as osp

import torch
from gymnasium import spaces
from mmengine import Config
from torch_geometric.data import Batch, Data

from .base import BaseModule
from .gdino import GDino

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
    def out_channels(self):
        pass


class EntityPassThrough(EntityEncoder):
    def __init__(self, model_config, obs_space):
        assert isinstance(obs_space, spaces.Graph)
        super().__init__(model_config, obs_space)

    @property
    def out_channels(self):
        # TODO: add global features
        out_channels = {
            "node_features": self._obs_space.node_space.shape,
            "edge_features": self._obs_space.edge_space.shape,
            "global_features": None,
        }
        return out_channels

    def forward(self, inputs):
        return inputs


class GDINOEncoder(EntityEncoder):
    def __init__(self, model_config, obs_space):
        assert isinstance(obs_space, spaces.Box)
        super().__init__(model_config, obs_space)

        self._model_config = model_config
        current_dir = osp.dirname(osp.abspath(__file__))
        gdino_cfg_file = osp.join(current_dir, model_config["gdino_cfg"])
        assert osp.exists(gdino_cfg_file)

        gdino_config = Config.fromfile(gdino_cfg_file).model
        if "num_queries" in model_config:
            gdino_config["num_queries"] = model_config["num_queries"]

        gdino_config.pop("language_model")
        gdino_config.pop("type")
        self._model = GDino(
            prompt_size=model_config["prompt_size"], **copy.deepcopy(gdino_config)
        )

    @property
    def out_channels(self):
        # cls_feat, bbox normalized coords  (cx, cy, w, h), stack depth
        x_shape = self._model.cls_features + 4 + 1
        return {
            "node_features": [x_shape],
            "edge_features": None,
            "global_features": None,
        }

    def forward(self, inputs):
        inputs = inputs.permute(0, 3, 1, 2)
        inputs = inputs.to(torch.float32) / 255.0
        # TODO: check normalization

        assert inputs.shape[1] % 3 == 0
        stack_depth = inputs.shape[1] // 3
        batch_size = inputs.shape[0]

        # This could be processed in parallel
        # but that would require more memory
        frame_nodes = []
        for i in range(stack_depth):
            frame = inputs[:, 3 * i : 3 * (i + 1)]
            outputs = self._model.forward(frame, mode="tensor")

            n_nodes = outputs["features"].shape[1]
            stack_feature = torch.tensor(
                [[i / stack_depth]], device=self.device
            ).expand(batch_size, n_nodes, -1)
            node_features = torch.cat(
                [outputs["features"], outputs["bboxes"], stack_feature],
                dim=2,
            )

            frame_nodes.append(node_features)

        frame_nodes = torch.stack(frame_nodes, dim=0)
        frame_nodes = frame_nodes.permute(1, 0, 2, 3)
        # Frame nodes has shape (batch_size, stack_depth, n_nodes, node_features)
        # The actual number of nodes should be stack_depth x n_nodes
        frame_nodes = frame_nodes.reshape(
            batch_size, stack_depth * frame_nodes.shape[2], -1
        )

        g_batch = []

        # This iterates over the batch dimension
        for node_features in frame_nodes:
            if self._config.get("add_edges", False):
                n_nodes = node_features.shape[0]
                edge_index = [[i, j] for i in range(n_nodes) for j in range(n_nodes)]
                edge_index = (
                    torch.tensor(edge_index, dtype=torch.long, device=self.device)
                    .t()
                    .contiguous()
                )

            else:
                edge_index = None

            g_batch.append(Data(x=node_features, edge_index=edge_index))

        batch = Batch.from_data_list(g_batch)
        return batch

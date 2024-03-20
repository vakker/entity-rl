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
        gdino_config.pop("language_model")
        gdino_config.pop("type")
        self._model = GDino(
            prompt_size=model_config["prompt_size"], **copy.deepcopy(gdino_config)
        )

    @property
    def out_channels(self):
        # cls_feat, bbox normalized coords  (cx, cy, w, h), stack depth
        x_shape = self._model.cls_features + 4 + 0
        return {
            "node_features": [x_shape],
            "edge_features": None,
            "global_features": None,
        }

    def forward(self, inputs):
        inputs = inputs.permute(0, 3, 1, 2)
        inputs = inputs.to(torch.float32) / 255.0
        # TODO: check normalization

        # Only take 3 channels for now
        inputs = inputs[:, :3]
        outputs = self._model.forward(inputs, mode="tensor")

        g_batch = []
        for sample in outputs:
            # TODO: add stack depth
            node_features = torch.cat([sample["features"], sample["bboxes"]], dim=1)
            g_batch.append(Data(x=node_features))

        batch = Batch.from_data_list(g_batch)
        return batch

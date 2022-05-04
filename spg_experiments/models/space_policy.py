# pylint: disable=undefined-loop-variable
from typing import Any, Tuple

import numpy as np
import torch
from ray.rllib.utils.typing import TensorType
from torch import nn
from torch_geometric.data import Batch, Data

from . import gnn
from .base import BasePolicy
from .space.arch import arch
from .space.space import Space


class SpaceGnnPolicy(BasePolicy):
    def _hidden_layers(self, input_dict):
        # TODO: fix stacking
        # print(input_dict["obs"].shape, input_dict["obs"].device)
        x = input_dict["obs"].permute(0, 3, 1, 2)[:, -3:]
        is_dummy = False

        if x.shape[0] == 32 and x.device != "cpu":
            x = torch.zeros(4, x.shape[1], x.shape[2], x.shape[3]).to(self.device)
            is_dummy = True

        loss, log = self._space_encoder(x, 0)

        g_batch = []

        data = zip(
            log["z_pres"][:, :, 0],
            log["z_what"],
            log["z_where"],
        )
        total_nodes = 0

        # TODO: the stacking is still a bit slow
        for z_pres_batch, z_what_batch, z_where_batch in data:
            batch_mask = z_pres_batch > 0.5
            z_what = z_what_batch[batch_mask]
            z_where = z_where_batch[batch_mask]

            x = torch.cat([z_what, z_where], dim=-1)

            n_nodes = x.shape[0]
            total_nodes += n_nodes
            if n_nodes != 0:
                edge_index = np.array(
                    [[i, j] for i in range(n_nodes) for j in range(n_nodes)]
                )
                edge_index = torch.tensor(edge_index).long()
                edge_index = torch.transpose(edge_index, 1, 0).to(self.device)
            else:
                x = torch.zeros((1, x.shape[1])).to(self.device)
                edge_index = torch.tensor(np.array([[0], [0]])).to(self.device)

            g_batch.append(Data(x=x, edge_index=edge_index))

        print("Total nodes:", total_nodes)
        batch = Batch.from_data_list(g_batch)
        features = self._gnn_encoder((batch.x, batch.edge_index, batch.batch))

        if is_dummy:
            features = features.tile((8, 1))

        return features

    def _create_hidden_layers(self, obs_space, model_config):
        in_channels = arch.z_what_dim + 4  # z_where_dim
        dims = model_config["custom_model_config"]["dims"]
        activation = model_config["custom_model_config"].get("activation")
        graph_conv = model_config["custom_model_config"].get(
            "graph_conv", "GATFeatures"
        )

        gnn_encoder = getattr(gnn, graph_conv)(
            n_input_features=in_channels,
            dims=dims,
            activation=activation,
        )

        self._space_encoder = Space()
        self._gnn_encoder = nn.Sequential(gnn_encoder, nn.Flatten())
        out_channels_all = gnn_encoder.out_channels

        return out_channels_all

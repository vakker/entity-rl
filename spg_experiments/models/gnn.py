import sys

import torch
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import MLP, GATv2Conv, GINConv, global_mean_pool

from .base import BasePolicy

module = sys.modules[__name__]


class GATFeatures(nn.Module):
    def __init__(self, n_input_features, dims, activation=None, norm=None, dropout=0.0):
        # pylint: disable=unused-argument
        super().__init__()

        if activation:
            self.act = getattr(nn, activation)()
        else:
            self.act = nn.ELU()

        # TODO: add normalization if needed
        # if norm:
        #     norm_layer = getattr(pyg_nn, norm)
        # else:
        #     norm_layer = nn.Identity

        convs = []
        in_channels = n_input_features
        for dim, heads in dims:
            convs.append(GATv2Conv(in_channels, dim, heads))
            in_channels = dim * heads

        self._convs = nn.ModuleList(convs)
        self.out_channels = in_channels

    def forward(self, inputs):
        x, edge_index, batch = inputs
        for conv in self._convs:
            x = self.act(conv(x, edge_index))

        x = global_mean_pool(x, batch)
        return x


class GINFeatures(torch.nn.Module):
    def __init__(self, n_input_features, dims, activation=None, norm=None, dropout=0.0):
        # pylint: disable=unused-argument
        super().__init__()

        if activation:
            self.act = getattr(nn, activation)()
        else:
            self.act = nn.ReLU()

        # TODO: add normalization if needed
        # if norm:
        #     norm_layer = getattr(pyg_nn, norm)
        # else:
        #     norm_layer = nn.Identity

        convs = []
        in_channels = n_input_features
        for dim in dims:
            mlp = MLP([in_channels, dim, dim], act=self.act)
            convs.append(GINConv(nn=mlp, train_eps=False))
            in_channels = dim

        self._convs = nn.ModuleList(convs)
        self.mlp = MLP(
            [in_channels, in_channels, in_channels],
            norm=None,
            dropout=dropout,
        )

        self.out_channels = in_channels

    def forward(self, inputs):
        x, edge_index, batch = inputs
        for conv in self._convs:
            x = self.act(conv(x, edge_index))

        x = global_mean_pool(x, batch)
        return self.mlp(x)


class GnnPolicy(BasePolicy):
    def _hidden_layers(self, input_dict):
        g_batch = []

        data = zip(
            input_dict["obs"]["x"].values,
            input_dict["obs"]["x"].lengths.long(),
            torch.transpose(input_dict["obs"]["edge_index"].values, 2, 1).long(),
            input_dict["obs"]["edge_index"].lengths.long(),
        )

        # TODO: the stacking is still a bit slow
        for x, x_len, edge_index, ei_len in data:
            if not x_len:
                x_len = 1
                ei_len = 1

            g_batch.append(Data(x=x[:x_len], edge_index=edge_index[:, :ei_len]))

        batch = Batch.from_data_list(g_batch)
        features = self._encoder((batch.x, batch.edge_index, batch.batch))

        return features

    def _create_hidden_layers(self, obs_space, model_config):
        in_channels = obs_space.original_space["x"].child_space.shape[0]

        self._n_input_size = in_channels
        graph_conv = model_config["custom_model_config"].get(
            "graph_conv", "GATFeatures"
        )

        gnn = getattr(module, graph_conv)(
            n_input_features=in_channels,
            **model_config["custom_model_config"].get("conv_config", {})
        )

        self._encoder = nn.Sequential(gnn, nn.Flatten())
        out_channels_all = gnn.out_channels

        return out_channels_all

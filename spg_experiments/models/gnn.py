import torch
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GINConv, global_mean_pool

from .base import BaseNetwork


class GINFeatures(nn.Module):
    def __init__(self, n_input_features, dims, activation=None):
        super().__init__()

        if activation:
            act = getattr(nn, activation)
        else:
            act = nn.ReLU

        convs = []
        in_channels = n_input_features
        for dim in dims:
            convs.append(
                GINConv(
                    nn.Sequential(
                        nn.Linear(in_channels, dim),
                        nn.BatchNorm1d(dim),
                        act(),
                        nn.Linear(dim, dim),
                        act(),
                    )
                )
            )
            in_channels = dim

        self._convs = nn.ModuleList(convs)

    def forward(self, inputs):
        x, edge_index, batch = inputs
        for conv in self._convs:
            x = conv(x, edge_index)

        x = global_mean_pool(x, batch)
        return x


class GnnNetwork(BaseNetwork):
    def _hidden_layers(self, input_dict):
        g_batch = []
        data = zip(
            input_dict["obs"]["x"].values,
            input_dict["obs"]["x"].lengths.long(),
            input_dict["obs"]["edge_index"].values,
            input_dict["obs"]["edge_index"].lengths.long(),
        )

        # TODO: the stacking is still a bit slow
        for x, x_len, edge_index, ei_len in data:
            if not x_len:
                x_len = 1
                ei_len = 1

            x = x[:x_len]
            edge_index = edge_index[:ei_len]
            edge_index = torch.transpose(edge_index, 1, 0).long()

            g_batch.append(Data(x=x, edge_index=edge_index))

        batch = Batch.from_data_list(g_batch)
        features = self._encoder((batch.x, batch.edge_index, batch.batch))

        return features

    def _create_hidden_layers(self, obs_space, model_config):
        in_channels = obs_space.original_space["x"].child_space.shape[0]

        self._n_input_size = in_channels
        dims = model_config["custom_model_config"]["dims"]
        activation = model_config["custom_model_config"].get("activation")

        gnn = GINFeatures(
            n_input_features=in_channels,
            dims=dims,
            activation=activation,
        )

        self._encoder = nn.Sequential(gnn, nn.Flatten())
        out_channels_all = dims[-1]

        return out_channels_all

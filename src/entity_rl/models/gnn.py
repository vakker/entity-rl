import sys

import torch
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import MLP, GATv2Conv, GINConv, global_mean_pool

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


class GINFeatures(nn.Module):
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


class GNNEncoder(nn.Module):
    def __init__(self, model_config, input_space):
        super().__init__()

        # FIXME:
        obs_space = input_space

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
        self._out_channeld = gnn.out_channels

    def forward(self, inputs):
        # TODO: refactor this
        return self._hidden_layers(inputs)

    def _hidden_layers(self, input_dict):
        g_batch = []

        if "edge_index" in input_dict["obs"]:
            edge_index = torch.transpose(
                input_dict["obs"]["edge_index"].values, 2, 1
            ).long()
            edge_index_len = input_dict["obs"]["edge_index"].lengths.long()

        else:
            edge_index = (None for _ in range(len(input_dict["obs_flat"])))
            edge_index_len = (None for _ in range(len(input_dict["obs_flat"])))

        data = zip(
            input_dict["obs"]["x"].values,
            input_dict["obs"]["x"].lengths.long(),
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
                    device=input_dict["obs_flat"].device,
                )

                j_idx = node_indices.tile((n_nodes,))
                i_idx = node_indices.repeat_interleave(n_nodes)
                edge_index = torch.stack([i_idx, j_idx], dim=0)

                ei_len = edge_index.shape[1]

            g_batch.append(Data(x=x[:x_len], edge_index=edge_index[:, :ei_len]))

        batch = Batch.from_data_list(g_batch)
        features = self._encoder((batch.x, batch.edge_index, batch.batch))

        return features

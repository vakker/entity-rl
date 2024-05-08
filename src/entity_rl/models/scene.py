import sys
from abc import abstractmethod

from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import MLP, GATv2Conv, aggr

from .base import BaseModule
from .slot_attention import SlotAttention

module = sys.modules[__name__]


def get_conv_layer(in_channels, config):
    layer_class = getattr(module, config["conv_name"])
    return layer_class(n_input_features=in_channels, **config["conv_config"])


# def get_aggr_layer(config):
#     layer_class = getattr(aggr, config['name'])
#     return layer_class(**config['config'])


class SceneEncoder(BaseModule):
    def __init__(self, model_config, obs_space):
        super().__init__()

        self._config = model_config
        self._obs_space = obs_space

    @abstractmethod
    def get_out_channels(self):
        pass

    @abstractmethod
    def forward(self, inputs):
        pass


class GATFeatures(BaseModule):
    def __init__(
        self,
        n_input_features,
        dims,
        activation=None,
        norm=None,
        dropout=0.0,
        aggr_layer="attn",
    ):
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
        self._out_channels = in_channels

        if aggr_layer == "attn":
            gate_nn = MLP([in_channels, 1])
            feat_nn = MLP([in_channels, in_channels])
            self._aggr = aggr.AttentionalAggregation(gate_nn, feat_nn)

        elif aggr_layer == "mean":
            self._aggr = aggr.MeanAggregation()

        else:
            self._aggr = None

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, inputs):
        x, edge_index, batch = inputs
        for conv in self._convs:
            x = self.act(conv(x, edge_index))

        if self._aggr is not None:
            x = self._aggr(x, batch)

        return x


# class GINFeatures(BaseModule):
#     def __init__(self, n_input_features, dims, activation=None, norm=None, dropout=0.0):
#         # pylint: disable=unused-argument
#         super().__init__()

#         if activation:
#             self.act = getattr(nn, activation)()
#         else:
#             self.act = nn.ReLU()

#         # TODO: add normalization if needed
#         # if norm:
#         #     norm_layer = getattr(pyg_nn, norm)
#         # else:
#         #     norm_layer = nn.Identity

#         convs = []
#         in_channels = n_input_features
#         for dim in dims:
#             mlp = MLP([in_channels, dim, dim], act=self.act)
#             convs.append(GINConv(nn=mlp, train_eps=False))
#             in_channels = dim

#         self._convs = nn.ModuleList(convs)
#         self.mlp = MLP(
#             [in_channels, in_channels, in_channels],
#             norm=None,
#             dropout=dropout,
#         )

#         self._out_channels = in_channels

#     @property
#     def out_channels(self):
#         return self._out_channels

#     def forward(self, inputs):
#         x, edge_index, batch = inputs
#         for conv in self._convs:
#             x = self.act(conv(x, edge_index))

#         x = global_mean_pool(x, batch)
#         return self.mlp(x)


class GNNEncoder(BaseModule):
    def __init__(self, model_config, input_space):
        super().__init__()

        assert len(input_space["node_features"]) == 1
        in_channels = input_space["node_features"][0]

        self._n_input_size = in_channels

        conv_layer = get_conv_layer(in_channels, model_config)

        self._encoder = nn.Sequential(conv_layer, nn.Flatten())
        self._out_channels = conv_layer.out_channels

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, inputs):
        assert isinstance(inputs, Batch)

        features = self._encoder((inputs.x, inputs.edge_index, inputs.batch))
        return features

    # def _hidden_layers(self, input_dict):
    #     g_batch = []

    #     if "edge_index" in input_dict["obs"]:
    #         edge_index = torch.transpose(
    #             input_dict["obs"]["edge_index"].values, 2, 1
    #         ).long()
    #         edge_index_len = input_dict["obs"]["edge_index"].lengths.long()

    #     else:
    #         edge_index = (None for _ in range(len(input_dict["obs_flat"])))
    #         edge_index_len = (None for _ in range(len(input_dict["obs_flat"])))

    #     data = zip(
    #         input_dict["obs"]["x"].values,
    #         input_dict["obs"]["x"].lengths.long(),
    #         edge_index,
    #         edge_index_len,
    #     )

    #     # TODO: the stacking is still a bit slow
    #     for x, x_len, edge_index, ei_len in data:
    #         if not x_len:
    #             x_len = 1
    #             ei_len = 1

    #         if edge_index is None:
    #             n_nodes = x_len

    #             # For refecence:
    #             # start_time = time.time()
    #             # edge_index = [
    #             #     torch.tensor([i, j], device=input_dict["obs_flat"].device)
    #             #     for i in range(n_nodes)
    #             #     for j in range(n_nodes)
    #             # ]
    #             # edge_index = torch.transpose(torch.stack(edge_index), 1, 0).long()
    #             # print("edge_index", time.time() - start_time)

    #             node_indices = torch.tensor(
    #                 range(n_nodes),
    #                 dtype=torch.long,
    #                 device=input_dict["obs_flat"].device,
    #             )

    #             j_idx = node_indices.tile((n_nodes,))
    #             i_idx = node_indices.repeat_interleave(n_nodes)
    #             edge_index = torch.stack([i_idx, j_idx], dim=0)

    #             ei_len = edge_index.shape[1]

    #         g_batch.append(Data(x=x[:x_len], edge_index=edge_index[:, :ei_len]))

    #     batch = Batch.from_data_list(g_batch)
    #     features = self._encoder((batch.x, batch.edge_index, batch.batch))

    #     return features


# class SlotAttnDecoderRef(nn.Module):
#     def __init__(self, model_config, input_space):
#         super().__init__()

#         # FIXME:
#         obs_space = input_space

#         dim = sum(s.shape[0] for k, s in obs_space.original_space.child_space.items())
#         self._n_input_size = dim
#         num_slots = model_config["custom_model_config"]["num_slots"]
#         hidden_dim = model_config["custom_model_config"]["hidden_dim"]

#         slot_attn = SlotAttention(
#             num_slots=num_slots,
#             dim=dim,
#             hidden_dim=hidden_dim,
#         )
#         self._encoder = nn.Sequential(slot_attn, nn.Flatten())
#         out_channels_all = num_slots * dim

#         return out_channels_all

#     def _hidden_layers(self, input_dict):
#         # NOTE: manual batching is used to work around stacking
#         # variable element size observations. This needs to be
#         # optimised, it's a significant bottleneck.
#         # This implementation is only for reference to test the more
#         # efficient implementation in the SlotAttnDecoder class.

#         features = []
#         for elements in input_dict["obs"].unbatch_all():
#             if elements:
#                 elem_tensor = []
#                 for elem in elements:
#                     elem_tensor.append(torch.cat([v for k, v in elem.items()]))
#                 elem_tensor = torch.stack(elem_tensor)

#             else:
#                 # Normally elements cannot be empty, but during
#                 # model init there's an empty sample for some reason
#                 # TODO: verify this
#                 elem_tensor = torch.zeros(
#                     1, self._n_input_size, device=input_dict["obs_flat"].device
#                 )

#             features.append(self._encoder(elem_tensor.unsqueeze(0)))

#         return torch.cat(features, dim=0)

#     def forward(self, inputs):
#         # TODO: refactor this
#         return self._hidden_layers(inputs)


class SlotAttnDecoder(BaseModule):
    def __init__(self, model_config, input_space):
        super().__init__()

        # in_channels = obs_space.original_space["x"].child_space.shape[0]
        in_channels = input_space["node_features"][0]

        self._n_input_size = in_channels
        num_slots = model_config["num_slots"]
        hidden_dim = model_config["hidden_dim"]
        final_act = model_config.get("final_act", True)

        slot_attn = SlotAttention(
            num_slots=num_slots,
            dim=in_channels,
            hidden_dim=hidden_dim,
            final_act=final_act,
        )
        self._encoder = nn.Sequential(slot_attn, nn.Flatten())
        self._out_channels = slot_attn.out_channels

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, inputs):
        # TODO: refactor this
        return self._hidden_layers(inputs)

    def _hidden_layers(self, inputs):
        assert isinstance(inputs, Batch)

        features = self._encoder((inputs.x, inputs.edge_index, inputs.batch))
        return features

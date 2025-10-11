import sys
from abc import abstractmethod
from typing import Callable, Optional, Tuple, Union

import torch
import torch_geometric.nn as pyg_nn
from pandas.core.computation.ops import Op
from torch import Tensor, nn
from torch_geometric.data import Batch
from torch_geometric.nn import MLP, GATv2Conv, GraphConv, SAGPooling, TopKPooling, aggr
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.inits import reset
from torch_geometric.nn.pool.connect import FilterEdges
from torch_geometric.nn.pool.select import Select, SelectOutput, SelectTopK
from torch_geometric.nn.pool.select.topk import topk
from torch_geometric.typing import OptTensor
from torch_geometric.utils import softmax

from entity_rl.utils import TicToc

from .base import BaseModule, hook_fn
from .slot_attention import SlotAttention

module = sys.modules[__name__]


def get_conv_layer(in_channels, config):
    conv_name = config.get("conv_name", "GATFeatures")
    if "conv_config" in config:
        conv_config = config["conv_config"]
    elif "conv" in config:
        conv_config = config["conv"]
    else:
        raise ValueError()

    layer_class = getattr(module, conv_name)
    return layer_class(n_input_features=in_channels, **conv_config)


def get_pooling_layer(in_channels, pooling_config):
    """
    Create pooling layer based on configuration.

    Args:
        in_channels: Number of input channels
        pooling_config: Dict with 'type' and optional 'params'

    Returns:
        Pooling layer instance
    """
    pooling_type = pooling_config["type"]
    pooling_params = pooling_config.get("params", {})

    # Support any pooling class from torch_geometric.nn
    try:
        if pooling_type == "SAGPooling":
            pooling_class = CustomSAGPooling
        else:
            pooling_class = getattr(pyg_nn, pooling_type)

        return pooling_class(in_channels=in_channels, **pooling_params)
    except AttributeError:
        raise ValueError(
            f"Unknown pooling type: {pooling_type}. "
            f"Must be a class from torch_geometric.nn"
        )


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


class NoOpSceneEncoder(BaseModule):
    def __init__(self, model_config, input_space):
        super().__init__()

        self._out_channels = input_space["node_features"][0]

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, inputs):
        # For no-op, just return the input features unchanged
        # inputs should be the entity features from the entity encoder
        return inputs


class GATFeatures(BaseModule):
    def __init__(
        self,
        n_input_features,
        dims,
        activation=None,
        norm=None,
        dropout=0.0,
        aggr_layer="mean",
        norm_layer=None,
    ):
        # pylint: disable=unused-argument
        super().__init__()

        if activation:
            self.act = getattr(nn, activation)()
        else:
            self.act = nn.ELU()

        # TODO: dropout?

        if norm_layer is None:
            norm_cls = nn.Identity

        elif norm_layer == "batch":
            norm_cls = pyg_nn.BatchNorm

        elif norm_layer == "layer":
            norm_cls = pyg_nn.LayerNorm

        elif norm_layer == "instance":
            raise ValueError("InstanceNorm makes no sense for single feature vectors.")
            # norm_cls = pyg_nn.InstanceNorm

        else:
            raise ValueError("Wrong norm_layer")

        # TODO: pre-norm?
        self._convs = nn.ModuleList()
        self._norms = nn.ModuleList()
        in_channels = n_input_features
        for dim, heads in dims:
            self._convs.append(GATv2Conv(in_channels, dim, heads))
            in_channels = dim * heads
            self._norms.append(norm_cls(in_channels))

        self._out_channels = in_channels

        if aggr_layer is None:
            self._aggr = None

        elif aggr_layer == "attn":
            # TODO: change act to LeakyReLU
            gate_nn = MLP([in_channels, 1])
            feat_nn = MLP([in_channels, in_channels])
            self._aggr = CustomAttentionalAggregation(gate_nn, feat_nn)

        elif aggr_layer == "mean":
            self._aggr = aggr.MeanAggregation()

        else:
            raise ValueError("Wrong aggr_layer")

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, inputs):
        x, edge_index, batch = inputs
        timer = TicToc(enabled=False)
        for conv, norm in zip(self._convs, self._norms):
            timer.tic("conv")
            x = self.act(norm(conv(x, edge_index)))
            timer.toc("conv")

        if self._aggr is not None:
            timer.tic("aggr")
            x = self._aggr(x, batch)
            timer.toc("aggr")

        timer.print_stats(title="GAT forward")
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

        # Add projection layer if configured
        projection_conf = model_config.get("projection", None)
        if projection_conf:
            layers = []

            # MLP projection
            mlp = MLP(
                in_channels=in_channels,
                num_layers=projection_conf["num_layers"],
                hidden_channels=projection_conf["out_channels"],
                out_channels=projection_conf["out_channels"],
                act="leakyrelu",
                norm=projection_conf.get("norm", None),
            )
            layers.append(mlp)

            # Post-normalization if specified
            post_norm = projection_conf.get("post_norm", None)
            if post_norm:
                if post_norm == "batch_norm":
                    layers.append(pyg_nn.BatchNorm(projection_conf["out_channels"]))
                elif post_norm == "layer_norm":
                    layers.append(pyg_nn.LayerNorm(projection_conf["out_channels"]))
                elif post_norm == "instance_norm":
                    layers.append(pyg_nn.InstanceNorm(projection_conf["out_channels"]))
                else:
                    raise ValueError(f"Unknown post_norm: {post_norm}")

            self.projection = nn.Sequential(*layers)
            in_channels = projection_conf["out_channels"]
        else:
            self.projection = None

        # Add pooling layer if configured
        pooling_config = model_config.get("pooling", None)
        if pooling_config:
            self.pooling = get_pooling_layer(in_channels, pooling_config)
        else:
            self.pooling = None

        conv_layer = get_conv_layer(in_channels, model_config)

        self._encoder = nn.Sequential(conv_layer, nn.Flatten())
        self._out_channels = conv_layer.out_channels

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, inputs):
        assert isinstance(inputs, Batch)

        x, edge_index, batch = inputs.x, inputs.edge_index, inputs.batch
        timer = TicToc(enabled=False)

        # Apply projection if configured
        if self.projection:
            timer.tic("projection")
            x = self.projection(x)
            timer.toc("projection")

        # Apply pooling before conv layers if configured
        if self.pooling:
            timer.tic("pooling")
            x, edge_index, _, batch, _, _ = self.pooling(x, edge_index, batch=batch)
            timer.toc("pooling")

        # Apply conv layers with potentially reduced graph
        timer.tic("conv")
        features = self._encoder((x, edge_index, batch))
        timer.toc("conv")

        timer.print_stats(title="GNNEncoder forward")
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


class CustomAttentionalAggregation(aggr.AttentionalAggregation):
    def __init__(
        self,
        gate_nn: torch.nn.Module,
        nn: Optional[torch.nn.Module] = None,
    ):
        super().__init__(gate_nn, nn)

        self.attention_acts = None

    def forward(
        self,
        x: Tensor,
        index: Optional[Tensor] = None,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
    ) -> Tensor:
        if self.gate_mlp is not None:
            gate = self.gate_mlp(x, batch=index, batch_size=dim_size)
        else:
            gate = self.gate_nn(x)

        if self.mlp is not None:
            x = self.mlp(x, batch=index, batch_size=dim_size)
        elif self.nn is not None:
            x = self.nn(x)

        gate = softmax(gate, index, ptr, dim_size, dim)
        self.attention_acts = gate.detach().cpu()
        return self.reduce(gate * x, index, ptr, dim_size, dim)


class CustomSAGPooling(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        ratio: Union[float, int] = 0.5,
        gate_channels: Optional[list] = None,
        proj_channels: Optional[list] = None,
        nonlinearity: Union[str, Callable] = "tanh",
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.ratio = ratio

        if gate_channels is None:
            gate_channels = [in_channels, 1]

        else:
            gate_channels = [in_channels] + gate_channels + [1]

        if proj_channels is not None:
            proj_channels = [in_channels] + proj_channels + [in_channels]
            self.mlp = MLP(channel_list=proj_channels, act="relu", norm=None)
            # self.mlp = MLP(channel_list=proj_channels, act="leakyrelu")

        else:
            self.mlp = None

        self.gate = MLP(channel_list=gate_channels, act="relu", norm=None)
        # self.gate.register_backward_hook(hook_fn)
        self.connect = FilterEdges()

        self.attention_acts = None
        self.perm = None

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.gate.reset_parameters()
        if self.mlp is not None:
            self.mlp.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: OptTensor = None,
        batch: OptTensor = None,
        dim_size: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor, OptTensor, OptTensor, Tensor, Tensor]:
        r"""Forward pass.

        Args:
            x (torch.Tensor): The node feature matrix.
            edge_index (torch.Tensor): The edge indices.
            edge_attr (torch.Tensor, optional): The edge features.
                (default: :obj:`None`)
            batch (torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each node to a specific example. (default: :obj:`None`)
        """
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        gate = self.gate(x, batch=batch)

        gate = softmax(gate, batch)
        self.attention_acts = gate.detach().cpu()

        node_index = topk(gate, self.ratio, batch)

        select_out = SelectOutput(
            node_index=node_index,
            num_nodes=x.size(0),
            cluster_index=torch.arange(node_index.size(0), device=x.device),
            num_clusters=node_index.size(0),
            weight=gate[node_index].squeeze(-1),
        )

        perm = select_out.node_index
        self.perm = perm.detach().cpu()
        score = select_out.weight
        assert score is not None

        x = x[perm]

        if self.mlp is not None:
            x = self.mlp(x, batch=batch)

        # We need this, otherwise we don't get gradients
        x = x * gate[perm]

        connect_out = self.connect(select_out, edge_index, edge_attr, batch)

        return (
            x,
            connect_out.edge_index,
            connect_out.edge_attr,
            connect_out.batch,
            perm,
            score,
        )

    def __repr__(self) -> str:
        if self.min_score is None:
            ratio = f"ratio={self.ratio}"
        else:
            ratio = f"min_score={self.min_score}"

        return (
            f"{self.__class__.__name__}({self.gnn.__class__.__name__}, "
            f"{self.in_channels}, {ratio}, multiplier={self.multiplier})"
        )

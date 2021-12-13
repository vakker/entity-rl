# pylint: disable=undefined-loop-variable
from typing import Dict, List

import gym
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from torch import nn
from torch_geometric.data import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool

from spg_experiments import graphs


class GnnNetwork(TorchModelV2, nn.Module):
    # pylint: disable=abstract-method

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        if not model_config.get("fcnet_hiddens"):
            raise ValueError("Config for fcnet_hiddens is required")
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        activation = self.model_config.get("fcnet_activation")
        hidden_sizes = self.model_config["fcnet_hiddens"]
        assert (
            len(hidden_sizes) > 0
        ), "Must provide at least 1 entry in `fcnet_hiddens`!"

        # Whether the last layer is the output of a Flattened (rather than
        # a n x (1,1) Conv2D).
        self.last_layer_is_flattened = False
        self._logits = None
        # Holds the current "base" output (before logits layer).
        self._features = None
        self.num_outputs = num_outputs
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.obs_space = obs_space

        self._create_model()

    def _create_model(self):
        in_channels = sum(
            [
                v.shape[0]
                for k, v in self.obs_space.original_space.child_space.spaces.items()
            ]
        )

        self.in_channels = in_channels
        features_dim = 512
        self._gnn = GINFeatures(n_input_features=in_channels, features_dim=features_dim)
        self.feat_size = features_dim

        # num_outputs defined. Use that to create an exact
        # `num_output`-sized mlp.
        if self.num_outputs:
            self._logits = SlimFC(
                features_dim,
                self.num_outputs,
                initializer=normc_initializer(0.01),
                activation_fn=None,
            )
        # num_outputs not known -> Flatten, then set self.num_outputs
        # to the resulting number of nodes.
        else:
            raise NotImplementedError()
            # self.last_layer_is_flattened = True
            # layers.append(nn.Flatten())
            # self.num_outputs = out_channels

        # Build the value layers
        self._value_branch = SlimFC(
            features_dim, 1, initializer=normc_initializer(0.01), activation_fn=None
        )

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):

        g_batch = []
        for elements in input_dict["obs"].unbatch_all():
            # print('elements')
            # for e in elements:
            #     print(e)
            nx_graph = graphs.build_fc_graph(
                elements, np.zeros((self.in_channels,), dtype=np.float32)
            )
            tg_graph = torch_geometric.utils.convert.from_networkx(nx_graph)
            g_batch.append(tg_graph)
        # sp = sum_params(self)
        # if np.isnan(sp):
        #     import ipdb
        #     ipdb.set_trace()

        # print('sum params', sp)
        # for name, param in self.named_parameters():
        #     print(name, torch.max(torch.abs(param.grad)))

        dl = DataLoader(g_batch, batch_size=len(g_batch), shuffle=False)
        data = next(iter(dl))
        data = data.to(input_dict["obs_flat"].device)
        self._features = self._gnn(data.x, data.edge_index, data.batch)

        # if torch.any(torch.isnan(self._features)):
        #     import ipdb; ipdb.set_trace()

        logits = self._logits(self._features)
        assert logits.shape[0] == input_dict["obs_flat"].shape[0]

        # if torch.any(torch.isnan(logits)):
        #     import ipdb; ipdb.set_trace()
        # print('model logits', logits)
        # if torch.any(torch.abs(logits) > 1000):
        #     import ipdb
        #     ipdb.set_trace()
        return logits, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"

        features = self._features

        # if torch.any(torch.isnan(features)):
        #     import ipdb; ipdb.set_trace()
        return self._value_branch(features).squeeze(1)

    def _hidden_layers(self, obs: TensorType) -> TensorType:
        res = self._layers(obs)
        return res

    # def parameters(self, *args, **kwargs):
    #     import ipdb; ipdb.set_trace()
    #     return super().parameters(self, *args, **kwargs)


class GINFeatures(nn.Module):
    def __init__(self, n_input_features: int, features_dim: int = 512):
        super().__init__()

        # self.act = F.selu
        self.act = F.relu

        nn1 = nn.Sequential(
            nn.Linear(n_input_features, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
        )
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(features_dim)

        nn2 = nn.Sequential(
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
        )
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(features_dim)

        nn3 = nn.Sequential(
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
        )
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(features_dim)

        nn4 = nn.Sequential(
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
        )
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(features_dim)

        nn5 = nn.Sequential(
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
        )
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(features_dim)

        self.fc1 = nn.Linear(features_dim, features_dim)

    def forward(self, x, edge_index, batch):
        x = self.act(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = self.act(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = self.act(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = self.act(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = self.act(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_mean_pool(x, batch)
        # x = self.fc1(x)
        x = torch.tanh(self.fc1(x))
        return x

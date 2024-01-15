# TODO: this has breaking changed since Ray 2.0
# pylint: disable=all
# pylint: disable=undefined-loop-variable

import numpy as np
import torch
from ray.rllib.algorithms.ppo import PPO as PPOTrainer
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.utils.numpy import convert_to_numpy
from torch import nn
from torch_geometric.data import Batch, Data

from . import gnn
from .base import BasePolicy
from .space.arch import arch
from .space.space import Space


# pylint: disable=attribute-defined-outside-init
class SpaceGnnPolicy(BasePolicy):
    def _hidden_layers(self, input_dict):
        # TODO: fix stacking
        # print(input_dict["obs"].shape, input_dict["obs"].device)
        x = input_dict["obs"].permute(0, 3, 1, 2)[:, -3:]
        # x = input_dict["obs"].permute(0, 3, 1, 2)
        is_dummy = False

        if x.shape[0] == 32 and x.device != "cpu":
            x = torch.zeros(4, x.shape[1], x.shape[2], x.shape[3]).to(self.device)
            is_dummy = True

        loss, log = self._space_encoder(x)
        self.space_loss = loss

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

    def on_global_var_update(self, global_vars):
        self._space_encoder.global_step = global_vars["timestep"]


# pylint: disable=abstract-method
class SpacePPOTorchPolicy(PPOTorchPolicy):
    def loss(self, model, dist_class, train_batch):
        total_loss = super().loss(model, dist_class, train_batch)

        if self.config.get("space_loss_coeff", 0.0) > 0.0:
            # TODO: move loss calculation here and use reduce_mean_valid instead
            total_loss += self.config["space_loss_coeff"] * model.space_loss
            model.tower_stats["mean_space_loss"] = model.space_loss

        return total_loss

    def extra_grad_info(self, train_batch):
        d_0 = super().extra_grad_info(train_batch)
        d_1 = convert_to_numpy(
            {
                "space_loss": torch.mean(
                    torch.stack(self.get_tower_stats("mean_space_loss"))
                )
            }
        )

        d_0.update(d_1)
        return d_0

    def on_global_var_update(self, global_vars):
        super().on_global_var_update(global_vars)
        if hasattr(self.model, "on_global_var_update"):
            self.model.on_global_var_update(global_vars)


# pylint: disable=abstract-method
class SpacePPOTrainer(PPOTrainer):
    _allow_unknown_configs = True

    def get_default_policy_class(self, config):
        return SpacePPOTorchPolicy

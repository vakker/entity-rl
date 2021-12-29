import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GINConv, global_mean_pool

from .base import BaseNetwork


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

    def forward(self, inputs):
        x, edge_index, batch = inputs
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
        features_dim = model_config["custom_model_config"]["features_dim"]

        gnn = GINFeatures(n_input_features=in_channels, features_dim=features_dim)

        self._encoder = nn.Sequential(gnn, nn.Flatten())
        out_channels_all = features_dim

        return out_channels_all

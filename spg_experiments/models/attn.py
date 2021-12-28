import torch
from torch import nn

from .base import BaseNetwork
from .slot_attention import SlotAttention


class AttnNetwork(BaseNetwork):
    def _hidden_layers(self, input_dict):
        # TODO: manual batching is used to work around stacking
        # variable element size observations. This needs to be
        # optimised if it's a bottlenack.

        features = []
        for elements in input_dict["obs"].unbatch_all():
            if elements:
                elem_tensor = []
                for elem in elements:
                    elem_tensor.append(torch.cat([v for k, v in elem.items()]))
                elem_tensor = torch.stack(elem_tensor)

            else:
                # Normally elements cannot be empty, but during
                # model init there's an empty sample for some reason
                # TODO: verify this
                elem_tensor = torch.zeros(
                    1, self._n_input_size, device=input_dict["obs_flat"].device
                )

            features.append(self._encoder(elem_tensor.unsqueeze(0)))

        return torch.cat(features, dim=0)

    def _create_hidden_layers(self, obs_space, model_config):
        dim = sum([s.shape[0] for k, s in obs_space.original_space.child_space.items()])
        self._n_input_size = dim
        num_slots = model_config["custom_model_config"]["num_slots"]
        hidden_dim = model_config["custom_model_config"]["hidden_dim"]

        slot_attn = SlotAttention(
            num_slots=num_slots,
            dim=dim,
            hidden_dim=hidden_dim,
        )
        self._encoder = nn.Sequential(slot_attn, nn.Flatten())
        out_channels_all = num_slots * dim

        return out_channels_all

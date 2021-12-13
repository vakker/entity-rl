# pylint: disable=undefined-loop-variable
from typing import Dict, List

import gym
import torch
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from torch import nn


class AttnNetwork(TorchModelV2, nn.Module):
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
        activation = self.activation
        layers = []
        in_channels = sum(
            [
                v.shape[0]
                for k, v in self.obs_space.original_space.child_space.spaces.items()
            ]
        )
        for out_channels in self.hidden_sizes:
            layers.append(
                SlimFC(
                    in_channels,
                    out_channels,
                    initializer=normc_initializer(0.01),
                    activation_fn=activation,
                )
            )
            in_channels = out_channels

        self._layers = nn.Sequential(*layers)
        self.feat_size = out_channels

        # num_outputs defined. Use that to create an exact
        # `num_output`-sized mlp.
        if self.num_outputs:
            self._logits = SlimFC(
                out_channels,
                self.num_outputs,
                initializer=normc_initializer(0.01),
                activation_fn=None,
            )
        # num_outputs not known -> Flatten, then set self.num_outputs
        # to the resulting number of nodes.
        else:
            self.last_layer_is_flattened = True
            layers.append(nn.Flatten())
            self.num_outputs = out_channels

        # Build the value layers
        self._value_branch = SlimFC(
            out_channels, 1, initializer=normc_initializer(0.01), activation_fn=None
        )

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        features = []
        for elements in input_dict["obs"].unbatch_all():
            if elements:
                elem_tensor = []
                for elem in elements:
                    elem_tensor.append(torch.cat([v for k, v in elem.items()]))
                elem_tensor = torch.stack(elem_tensor)
                print(elem_tensor)
                features.append(
                    torch.mean(self._layers(elem_tensor), dim=0, keepdim=True)
                )
            else:
                features.append(torch.zeros((1, self.feat_size)))

        self._features = torch.cat(features, dim=0)

        logits = self._logits(self._features)
        assert logits.shape[0] == input_dict["obs_flat"].shape[0]

        return logits, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"

        features = self._features
        return self._value_branch(features).squeeze(1)

    def _hidden_layers(self, obs: TensorType) -> TensorType:
        res = self._layers(obs)
        return res

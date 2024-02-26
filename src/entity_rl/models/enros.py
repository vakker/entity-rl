from typing import Dict, List

import gymnasium as gym
import torch
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from torch import nn

from . import combined
from .base import BaseModule


class Encoder(nn.Module):
    def __init__(self, model_config, input_shape):
        super().__init__()

        # The encoder needs to resolve the structure.
        # It can be either entity + scene or a combined encoder.
        # Either there's a combined encoder, or separate entity and scene encoders.

        self._stages = nn.ModuleList()
        if "combined" in model_config:
            encoder_name = model_config["combined"]["name"]
            encoder_config = model_config["combined"]["config"]
            encoder = getattr(combined, encoder_name)
            self._stages.append(encoder(encoder_config, input_shape))

        else:
            encoder_name = model_config["entity"]["name"]
            encoder_config = model_config["entity"]["config"]
            encoder = getattr(combined, encoder_name)
            self._stages.append(encoder(encoder_config, input_shape))

            input_shape = self._stages[-1].out_channels
            encoder_name = model_config["scene"]["name"]
            encoder_config = model_config["scene"]["config"]
            encoder = getattr(combined, encoder_name)
            self._stages.append(encoder(encoder_config, input_shape))

    @property
    def out_channels(self):
        return self._stages[-1].out_channels

    def forward(self, inputs):
        for stage in self._stages:
            inputs = stage(inputs)

        return inputs


# class Heads(nn.Module):
#     def __init__(self, model_config, in_channels, out_channels):
#         super().__init__()

#         # This could work with SlimConv2d, but it's unclear if it's actually faster
#         # Build the policy layers
#         # NOTE: tanh is still added on top of this later
#         self._policy = SlimFC(
#             in_channels,
#             out_channels,
#             # initializer=normc_initializer(0.01), # ?
#             activation_fn=None,
#         )

#         # Build the value layers
#         self._vf = SlimFC(
#             in_channels,
#             1,
#             # initializer=normc_initializer(0.01), # ?
#             activation_fn=None,
#         )

#         self._out_channels = out_channels

#     @property
#     def out_channels(self):
#         # TODO: this should be separate for policy and value
#         return self._out_channels

#     def forward_policy(self, inputs):
#         # NOTE: this is a trick to avoid issues with the action distribution
#         logits = torch.tanh(self._policy(inputs)) * 10
#         return logits

#     def forward_vf(self, inputs):
#         return self._vf(inputs).squeeze(1)


class ENROSPolicy(TorchModelV2, BaseModule):
    # pylint: disable=abstract-method,unused-argument
    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        **custom_model_kwargs: dict,
    ):
        assert num_outputs is not None, "num_outputs must be set"

        TorchModelV2.__init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
        )
        BaseModule.__init__(self)

        model_config = model_config["custom_model_config"]
        self._obs_space = obs_space
        input_shape = obs_space.shape
        self._encoder = Encoder(model_config["encoder"], input_shape)

        # ####### Heads
        # This could work with SlimConv2d, but it's unclear if it's actually faster
        # Build the policy layers
        # NOTE: tanh is still added on top of this later
        self._policy = SlimFC(
            self._encoder.out_channels,
            num_outputs,
            # initializer=normc_initializer(0.01), # ?
            activation_fn=None,
        )
        self._vf = SlimFC(
            self._encoder.out_channels,
            1,
            # initializer=normc_initializer(0.01), # ?
            activation_fn=None,
        )

        self._features = None

        # TODO: add this as some logging option
        # print('#################')
        # print(self)
        # print(count_parameters(self))

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ):
        self._features = self._encoder(input_dict["obs"])
        logits = self._policy(self._features)

        # NOTE: this is a trick to avoid issues with the action distribution
        logits = torch.tanh(logits) * 10
        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"

        # squeeze(1) is needed because the value function is expected to be just
        # a single number per batch element, and not B x 1
        return self._vf(self._features).squeeze(1)

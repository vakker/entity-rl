from abc import ABC, abstractmethod
from typing import Dict, List

import gymnasium as gym
import torch
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from torch import nn


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class BaseModule(nn.Module, ABC):
    @property
    def device(self):
        return next(self.parameters()).device


class BasePolicy(TorchModelV2, BaseModule, ABC):
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
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        BaseModule.__init__(self)

        self.obs_space = obs_space

        self._encoder = None
        out_channels_all = self._create_hidden_layers(obs_space, self.model_config)

        # TODO
        # Post FC net config.
        # post_fcnet_hiddens = model_config.get("post_fcnet_hiddens", [])
        # post_fcnet_activation = get_activation_fn(
        #     model_config.get("post_fcnet_activation"), framework="torch"
        # )

        self._logits = None
        # num_outputs defined. Use that to create an exact
        # `num_output`-sized MLP.
        if num_outputs:
            self._logits = SlimFC(
                out_channels_all,
                num_outputs,
                initializer=normc_initializer(0.01),
                activation_fn=None,
            )
            self.num_outputs = num_outputs

        # num_outputs not known -> Flatten, then set self.num_outputs
        # to the resulting number of nodes.
        else:
            self.num_outputs = out_channels_all

        # Build the value layers
        self._value_branch = SlimFC(
            out_channels_all,
            1,
            initializer=normc_initializer(0.01),
            activation_fn=None,
        )

        # Holds the current "base" output (before logits layer).
        self._features = None

        # print('#################')
        # print(self)
        # print(count_parameters(self))

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        self._features = self._hidden_layers(input_dict)

        if self._logits is None:
            return self._features, state

        logits = self._logits(self._features)
        logits = torch.tanh(logits) * 10
        return logits, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"

        return self._value_branch(self._features).squeeze(1)

    @abstractmethod
    def _hidden_layers(self, input_dict):
        pass

    @abstractmethod
    def _create_hidden_layers(self, obs_space, model_config):
        pass

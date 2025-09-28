import torch
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from torch import nn

from . import combined, entity, scene
from .base import BaseModule, get_num_params


class Encoder(BaseModule):
    def __init__(self, model_config, obs_space):
        super().__init__()

        self.include_agent_pos = model_config.get("include_agent_pos", False)

        # The encoder needs to resolve the structure.
        # It can be either entity + scene or a combined encoder.
        # Either there's a combined encoder, or separate entity and scene encoders.

        self._stages = nn.ModuleList()
        if "combined" in model_config:
            encoder_name = model_config["combined"]["name"]
            encoder_config = model_config["combined"].get("config")
            encoder = getattr(combined, encoder_name)
            self._stages.append(encoder(encoder_config, obs_space))

        else:
            encoder_name = model_config["entity"]["name"]
            encoder_config = model_config["entity"].get("config")
            encoder = getattr(entity, encoder_name)
            self._stages.append(encoder(encoder_config, obs_space))

            input_shape = self._stages[-1].out_channels
            encoder_name = model_config["scene"]["name"]
            encoder_config = model_config["scene"].get("config")
            encoder = getattr(scene, encoder_name)
            self._stages.append(encoder(encoder_config, input_shape))

    @property
    def out_channels(self):
        base_channels = self._stages[-1].out_channels
        # Add 4 channels for agent position only if include_agent_pos is True
        return base_channels + (4 if self.include_agent_pos else 0)

    def forward(self, inputs, agent_pos=None):
        for stage in self._stages:
            inputs = stage(inputs)

        if self.include_agent_pos:
            if agent_pos is None:
                raise ValueError("agent_pos is required when include_agent_pos=True")
            inputs = torch.cat([inputs, agent_pos], dim=1)
        # If include_agent_pos=False, ignore agent_pos and don't concatenate

        return inputs

    @property
    def num_params(self):
        num_params = {}
        for stage in self._stages:
            num_params[stage.__class__.__name__] = get_num_params(stage)

        return num_params


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
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        **custom_model_kwargs,
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

        self.use_amp = model_config.get("amp", False)
        model_config = model_config["custom_model_config"]

        if hasattr(obs_space, "original_space"):
            obs_space = obs_space.original_space

        self._obs_space = obs_space
        self._encoder = Encoder(model_config["encoder"], obs_space)

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
            # 3,
            # initializer=normc_initializer(0.01), # ?
            activation_fn=None,
        )

        self._features = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        with torch.autocast(device_type="cuda", enabled=self.use_amp):
            agent_pos = input_dict.get("agent_pos", None)
            self._features = self._encoder(input_dict["obs"], agent_pos=agent_pos)
            logits = self._policy(self._features)

            # NOTE: this is a trick to avoid issues with the action distribution
            logits = torch.tanh(logits) * 10
        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"

        # squeeze(1) is needed because the value function is expected to be just
        # a single number per batch element, and not B x 1
        with torch.autocast(device_type="cuda", enabled=self.use_amp):
            return self._vf(self._features).squeeze(1)

    @property
    def num_params(self):
        num_params = self._encoder.num_params
        num_params["policy"] = get_num_params(self._policy)
        num_params["vf"] = get_num_params(self._vf)
        return num_params

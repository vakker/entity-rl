import sys

import torch
from ray.rllib.models.torch.misc import SlimConv2d
from torch import nn

from .base import BaseModule

module = sys.modules[__name__]


MOBILENET_INPUT_SHAPE = (3, 224, 224)

# TODO: maybe subclass from this
# class CombinedEncoder(nn.Module, ABC):
#     def __init__(self, model_config, obs_space):
#         assert isinstance(obs_space, spaces.Box)

#         super().__init__()

#         self._obs_space = obs_space

#     @abstractmethod
#     def get_out_channels(self):
#         pass

#     @abstractmethod
#     def forward(self, inputs):
#         pass


class CNNEncoder(BaseModule):
    """Generic CNN encoder."""

    def __init__(self, model_config, obs_space):
        super().__init__()

        input_shape = obs_space.shape
        activation = model_config.get("conv_activation")
        filters = model_config["conv_filters"]
        assert len(filters) > 0, "Must provide at least 1 entry in `conv_filters`!"

        layers = []
        in_channels = input_shape[-1]

        for out_channels, kernel, stride in filters:
            padding = [k // 2 for k in kernel]
            layers.append(
                SlimConv2d(
                    in_channels,
                    out_channels,
                    kernel,
                    stride,
                    padding,
                    activation_fn=activation,
                )
            )
            in_channels = out_channels

        layers.append(nn.Flatten())
        self._convs = nn.Sequential(*layers)

        self._out_channels = None
        self._input_shape = input_shape

    def forward(self, inputs):
        # Obs is generally B x H x W x C, but CNNs expect B x C x H x W.
        # Normalize
        # FIXME: check gdino normalization
        inputs = (inputs.to(torch.float32) / 128.0) - 1.0
        return self._convs(inputs.permute(0, 3, 1, 2))

    @property
    def out_channels(self):
        if self._out_channels is None:
            dummy_in = torch.zeros(1, *self._input_shape)
            dummy_out = self._convs(dummy_in.permute(0, 3, 1, 2))
            self._out_channels = dummy_out.shape[1]

        return self._out_channels

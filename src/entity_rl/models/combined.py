import sys
from typing import List

import torch

# from ray.rllib.core.models.base import ENCODER_OUT, Encoder
# from ray.rllib.core.models.configs import ModelConfig
# from ray.rllib.core.models.torch.base import TorchModel
from ray.rllib.models.torch.misc import SlimConv2d
from ray.rllib.utils.typing import ModelConfigDict
from torch import nn

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


# class MobileNetV2EncoderConfig(ModelConfig):
#     # MobileNet v2 has a flat output with a length of 1000.
#     output_dims = (1000,)
#     freeze = True

#     def build(self, framework):
#         assert framework == "torch", "Unsupported framework `{}`!".format(framework)
#         return MobileNetV2Encoder(self)


# class MobileNetV2Encoder(TorchModel, Encoder):
#     """A MobileNet v2 encoder for RLlib."""

#     def __init__(self, config):
#         super().__init__(config)
#         self.net = torch.hub.load(
#             "pytorch/vision:v0.6.0", "mobilenet_v2", pretrained=True
#         )
#         if config.freeze:
#             # We don't want to train this encoder, so freeze its parameters!
#             for p in self.net.parameters():
#                 p.requires_grad = False

#     def _forward(self, input_dict, **kwargs):
#         return {ENCODER_OUT: (self.net(input_dict["obs"]))}


class CNNEncoder(nn.Module):
    """Generic CNN encoder."""

    def __init__(
        self,
        model_config: ModelConfigDict,
        input_shape: List[int],
    ):
        super().__init__()

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
        return self._convs(inputs.permute(0, 3, 1, 2))

    @property
    def out_channels(self):
        if self._out_channels is None:
            dummy_in = torch.zeros(1, *self._input_shape)
            dummy_out = self._convs(dummy_in.permute(0, 3, 1, 2))
            self._out_channels = dummy_out.shape[1]

        return self._out_channels

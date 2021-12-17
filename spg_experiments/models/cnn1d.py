# pylint: disable=undefined-loop-variable
from typing import Any, Dict, List, Tuple

import gym
import numpy as np
import torch
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from torch import nn


def get_out_size(in_size, padding, kernel_size, stride=1, dilation=1):
    return 1 + (in_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride


def create_branch(filters, activation, space):
    layers = []
    w, in_channels = space.shape
    in_size = w
    for out_channels, kernel, stride in filters:
        padding = 0
        layers.append(
            SlimConv1d(
                in_channels,
                out_channels,
                kernel,
                stride,
                padding,
                activation_fn=activation,
            )
        )
        in_channels = out_channels
        out_size = get_out_size(in_size, padding, kernel, stride)
        in_size = out_size

    layers.append(nn.Flatten())

    out_channels *= out_size
    out_size = 1

    return nn.Sequential(*layers), out_size, out_channels


def same_padding_1d(
    in_size: int,
    filter_size: int,
    stride_size: int,
) -> (int, Tuple[int, int]):
    """Note: Padding is added to match TF conv2d `same` padding. See
    www.tensorflow.org/versions/r0.12/api_docs/python/nn/convolution

    Args:
        in_size (tuple): Rows (Height), Column (Width) for input
        stride_size (Union[int,Tuple[int, int]]): Rows (Height), column (Width)
            for stride. If int, height == width.
        filter_size (tuple): Rows (Height), column (Width) for filter

    Returns:
        padding (tuple): For input into torch.nn.ZeroPad2d.
        output (tuple): Output shape after padding and convolution.
    """
    in_width = in_size
    filter_width = filter_size
    stride_width = stride_size

    out_width = int(np.ceil(in_width / stride_width))

    pad_along_width = int(((out_width - 1) * stride_width + filter_width - in_width))
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    padding = (pad_left, pad_right)
    output = out_width
    return padding, output


class SlimConv1d(nn.Module):
    """Simple mock of tf.slim Conv2d"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int,
        stride: int,
        padding: Tuple[int, int],
        # Defaulting these to nn.[..] will break soft torch import.
        initializer: Any = "default",
        activation_fn: Any = "default",
        bias_init: float = 0,
    ):
        """Creates a standard Conv2d layer, similar to torch.nn.Conv2d

        Args:
            in_channels(int): Number of input channels
            out_channels (int): Number of output channels
            kernel (Union[int, Tuple[int, int]]): If int, the kernel is
                a tuple(x,x). Elsewise, the tuple can be specified
            stride (Union[int, Tuple[int, int]]): Controls the stride
                for the cross-correlation. If int, the stride is a
                tuple(x,x). Elsewise, the tuple can be specified
            padding (Union[int, Tuple[int, int]]): Controls the amount
                of implicit zero-paddings during the conv operation
            initializer (Any): Initializer function for kernel weights
            activation_fn (Any): Activation function at the end of layer
            bias_init (float): Initalize bias weights to bias_init const
        """
        super().__init__()
        layers = []
        # Padding layer.
        if padding:
            layers.append(nn.ConstantPad1d(padding, 0))
        # Actual Conv1D layer (including correct initialization logic).
        conv = nn.Conv1d(in_channels, out_channels, kernel, stride)
        if initializer:
            if initializer == "default":
                initializer = nn.init.xavier_uniform_
            initializer(conv.weight)
        nn.init.constant_(conv.bias, bias_init)
        layers.append(conv)
        # Activation function (if any; default=ReLu).
        if isinstance(activation_fn, str):
            if activation_fn == "default":
                activation_fn = nn.ReLU
            else:
                activation_fn = get_activation_fn(activation_fn, "torch")
        if activation_fn is not None:
            layers.append(activation_fn())
        # Put everything in sequence.
        self._model = nn.Sequential(*layers)

    def forward(self, x: TensorType) -> TensorType:
        if x.dim() == 4:
            x = torch.squeeze(x, dim=2)
        return self._model(x)


class Cnn1DNetwork(TorchModelV2, nn.Module):
    # pylint: disable=abstract-method

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        if not model_config.get("conv_filters"):
            raise ValueError("Config for conv_filters is required")

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        activation = self.model_config.get("conv_activation")
        filters = self.model_config["conv_filters"]
        assert len(filters) > 0, "Must provide at least 1 entry in `conv_filters`!"

        # TODO
        # Post FC net config.
        # post_fcnet_hiddens = model_config.get("post_fcnet_hiddens", [])
        # post_fcnet_activation = get_activation_fn(
        #     model_config.get("post_fcnet_activation"), framework="torch"
        # )

        branches = {}
        out_channels_all = 0
        for obs_name, space in obs_space.original_space.spaces.items():
            branch, out_size, out_channels = create_branch(filters, activation, space)
            assert out_size == 1

            out_channels_all += out_channels
            branches[obs_name] = branch

            # TODO: add fix for inconsistent size, i.e. a final linear
            # layer or something that makes the out_size consistent.
            # Currently the flatten makes it consistent anyway.

        self._convs = nn.ModuleDict(branches)

        self._logits = None
        # num_outputs defined. Use that to create an exact
        # `num_output`-sized (1)-Conv1D.
        # TODO: add post_fcnet_hiddens
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
        return logits, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"

        return self._value_branch(self._features).squeeze(1)

    def _hidden_layers(self, input_dict):
        features = []
        for obs_name, obs in input_dict["obs"].items():
            features.append(self._convs[obs_name](obs.permute(0, 2, 1)))
        return torch.cat(features, dim=1)

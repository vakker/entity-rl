# pylint: disable=undefined-loop-variable
from typing import Any, Dict, List, Tuple

import gym
import numpy as np
import torch
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.utils import get_activation_fn, get_filter_config
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from torch import nn


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

    out_width = np.ceil(float(in_width) / float(stride_width))

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
    # pylint: disable=abstract-method,too-many-locals
    # pylint: disable=too-many-branches,too-many-statements

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):

        if not model_config.get("conv_filters"):
            model_config["conv_filters"] = get_filter_config(obs_space.shape)

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        activation = self.model_config.get("conv_activation")
        filters = self.model_config["conv_filters"]
        assert len(filters) > 0, "Must provide at least 1 entry in `conv_filters`!"

        # Post FC net config.
        post_fcnet_hiddens = model_config.get("post_fcnet_hiddens", [])
        post_fcnet_activation = get_activation_fn(
            model_config.get("post_fcnet_activation"), framework="torch"
        )

        no_final_linear = self.model_config.get("no_final_linear")
        vf_share_layers = self.model_config.get("vf_share_layers")

        # Whether the last layer is the output of a Flattened (rather than
        # a n x (1) Conv1D).
        self.last_layer_is_flattened = False
        self._logits = None

        layers = []
        (w, in_channels) = obs_space.shape

        in_size = w
        for out_channels, kernel, stride in filters[:-1]:
            padding, out_size = same_padding_1d(in_size, kernel, stride)
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
            in_size = out_size

        out_channels, kernel, stride = filters[-1]

        # No final linear: Last layer has activation function and exits with
        # num_outputs nodes (this could be a 1x1 conv or a FC layer, depending
        # on `post_fcnet_...` settings).
        if no_final_linear and num_outputs:
            out_channels = out_channels if post_fcnet_hiddens else num_outputs
            layers.append(
                SlimConv1d(
                    in_channels,
                    out_channels,
                    kernel,
                    stride,
                    None,  # padding=valid
                    activation_fn=activation,
                )
            )

            # Add (optional) post-fc-stack after last Conv2D layer.
            layer_sizes = post_fcnet_hiddens[:-1] + (
                [num_outputs] if post_fcnet_hiddens else []
            )
            for i, out_size in enumerate(layer_sizes):
                layers.append(
                    SlimFC(
                        in_size=out_channels,
                        out_size=out_size,
                        activation_fn=post_fcnet_activation,
                        initializer=normc_initializer(1.0),
                    )
                )
                out_channels = out_size

        # Finish network normally (w/o overriding last layer size with
        # `num_outputs`), then add another linear one of size `num_outputs`.
        else:
            layers.append(
                SlimConv1d(
                    in_channels,
                    out_channels,
                    kernel,
                    stride,
                    None,  # padding=valid
                    activation_fn=activation,
                )
            )

            # num_outputs defined. Use that to create an exact
            # `num_output`-sized (1,1)-Conv2D.
            if num_outputs:
                in_size = np.ceil((in_size - kernel) / stride)
                padding, _ = same_padding_1d(in_size, 1, 1)
                if post_fcnet_hiddens:
                    layers.append(nn.Flatten())
                    in_size = out_channels
                    # Add (optional) post-fc-stack after last Conv2D layer.
                    for i, out_size in enumerate(post_fcnet_hiddens + [num_outputs]):
                        layers.append(
                            SlimFC(
                                in_size=in_size,
                                out_size=out_size,
                                activation_fn=post_fcnet_activation
                                if i < len(post_fcnet_hiddens) - 1
                                else None,
                                initializer=normc_initializer(1.0),
                            )
                        )
                        in_size = out_size
                    # Last layer is logits layer.
                    self._logits = layers.pop()

                else:
                    self._logits = SlimConv1d(
                        out_channels,
                        num_outputs,
                        1,
                        1,
                        padding,
                        activation_fn=None,
                    )

            # num_outputs not known -> Flatten, then set self.num_outputs
            # to the resulting number of nodes.
            else:
                self.last_layer_is_flattened = True
                layers.append(nn.Flatten())

        self._convs = nn.Sequential(*layers)

        # If our num_outputs still unknown, we need to do a test pass to
        # figure out the output dimensions. This could be the case, if we have
        # the Flatten layer at the end.
        if self.num_outputs is None:
            # Create a B=1 dummy sample and push it through out conv-net.
            dummy_in = (
                torch.from_numpy(self.obs_space.sample())
                .permute(1, 0)
                .unsqueeze(0)
                .float()
            )
            dummy_out = self._convs(dummy_in)
            self.num_outputs = dummy_out.shape[1]

        # Build the value layers
        self._value_branch_separate = self._value_branch = None
        if vf_share_layers:
            self._value_branch = SlimFC(
                out_channels, 1, initializer=normc_initializer(0.01), activation_fn=None
            )
        else:
            vf_layers = []
            (w, in_channels) = obs_space.shape
            in_size = w
            for out_channels, kernel, stride in filters[:-1]:
                padding, out_size = same_padding_1d(in_size, kernel, stride)
                vf_layers.append(
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
                in_size = out_size

            out_channels, kernel, stride = filters[-1]
            vf_layers.append(
                SlimConv1d(
                    in_channels,
                    out_channels,
                    kernel,
                    stride,
                    None,
                    activation_fn=activation,
                )
            )

            vf_layers.append(
                SlimConv1d(
                    in_channels=out_channels,
                    out_channels=1,
                    kernel=1,
                    stride=1,
                    padding=None,
                    activation_fn=None,
                )
            )
            self._value_branch_separate = nn.Sequential(*vf_layers)

        # Holds the current "base" output (before logits layer).
        self._features = None

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        self._features = input_dict["obs"].float()
        # Permuate b/c data comes in as [B, dim, dim, channels]:
        self._features = self._features.permute(0, 2, 1)
        conv_out = self._convs(self._features)
        # Store features to save forward pass when getting value_function out.
        if not self._value_branch_separate:
            self._features = conv_out

        if not self.last_layer_is_flattened:
            if self._logits:
                conv_out = self._logits(conv_out)
            if len(conv_out.shape) == 3:
                if conv_out.shape[2] != 1:
                    conv_filters = self.model_config["conv_filters"]
                    num_outputs = self.num_outputs
                    shape = list(conv_out.shape)
                    raise ValueError(
                        f"Given `conv_filters` ({conv_filters}) do not"
                        f"result in a [B, {num_outputs} "
                        f"(`num_outputs`), 1] shape (but in {shape})! Please "
                        "adjust your Conv1D stack such that the last 1 dim "
                        "is 1."
                    )
                logits = conv_out.squeeze(2)
                logits = logits.squeeze(1)
            else:
                logits = conv_out
            return logits, state
        return conv_out, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        if self._value_branch_separate:
            value = self._value_branch_separate(self._features)
            value = value.squeeze(2)
            return value.squeeze(1)

        if not self.last_layer_is_flattened:
            features = self._features.squeeze(2)
        else:
            features = self._features
        return self._value_branch(features).squeeze(1)

    def _hidden_layers(self, obs: TensorType) -> TensorType:
        res = self._convs(obs.permute(0, 2, 1))  # switch to channel-major
        res = res.squeeze(2)
        return res

from typing import Any, Dict, List, Tuple, Union

import gym
import numpy as np
import torch
from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
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

    pad_along_width = int(
        ((out_width - 1) * stride_width + filter_width - in_width))
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    padding = (pad_left, pad_right)
    output = out_width
    return padding, output


# def ortho_init_1d(scale=1.0):
#     """
#     Orthogonal initialization for the policy weights
#     :param scale: (float) Scaling factor for the weights.
#     :return: (function) an initialization function for the weights
#     """

#     # _ortho_init(shape, dtype, partition_info=None)
#     def _ortho_init(shape, *_, **_kwargs):
#         """Intialize weights as Orthogonal matrix.
#         Orthogonal matrix initialization [1]_. For n-dimensional shapes where
#         n > 2, the n-1 trailing axes are flattened. For convolutional layers, this
#         corresponds to the fan-in, so this makes the initialization usable for
#         both dense and convolutional layers.
#         References
#         ----------
#         .. [1] Saxe, Andrew M., James L. McClelland, and Surya Ganguli.
#                "Exact solutions to the nonlinear dynamics of learning in deep
#                linear
#         """
#         # lasagne ortho init for tf
#         shape = tuple(shape)
#         flat_shape = (numpy.prod(shape[:-1]), shape[-1])

#         gaussian_noise = numpy.random.normal(0.0, 1.0, flat_shape)
#         u, _, v = numpy.linalg.svd(gaussian_noise, full_matrices=False)
#         weights = u if u.shape == flat_shape else v  # pick the one with the correct shape
#         weights = weights.reshape(shape)
#         return (scale * weights[:shape[0], :shape[1]]).astype(numpy.float32)

#     return _ortho_init


class CustomFC(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        self.torch_sub_model = TorchFC(obs_space, action_space, num_outputs,
                                       model_config, name)

    def forward(self, input_dict, state, seq_lens):
        fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
        return fc_out, []

    def value_function(self):
        return torch.reshape(self.torch_sub_model.value_function(), [-1])

    def sum_params(self):
        s = 0
        for p in self.parameters():
            s += p.sum()
        return s.item()


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
            bias_init: float = 0):
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


class VisionNetwork1D(TorchModelV2, nn.Module):
    """Generic vision network."""
    def __init__(self, obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str):
        if not model_config.get("conv_filters"):
            raise ValueError("Config for conv_filters is required")
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        activation = self.model_config.get("conv_activation")
        filters = self.model_config["conv_filters"]
        assert len(filters) > 0,\
            "Must provide at least 1 entry in `conv_filters`!"

        # Whether the last layer is the output of a Flattened (rather than
        # a n x (1,1) Conv2D).
        self.last_layer_is_flattened = False
        self._logits = None

        layers = []
        # FIXME add stacking here
        (w, in_channels) = obs_space.shape
        in_size = w
        for i, (out_channels, kernel, stride) in enumerate(filters):
            padding, out_size = same_padding_1d(in_size, kernel, stride)
            layers.append(
                SlimConv1d(in_channels,
                           out_channels,
                           kernel,
                           stride,
                           None if i == (len(filters) - 1) else padding,
                           activation_fn=activation))
            in_channels = out_channels
            in_size = out_size

        # num_outputs defined. Use that to create an exact
        # `num_output`-sized (1,1)-Conv2D.
        if num_outputs:
            in_size = np.ceil((in_size - kernel) / stride)

            padding, _ = same_padding_1d(in_size, 1, 1)
            self._logits = SlimConv1d(out_channels,
                                      num_outputs,
                                      1,
                                      1,
                                      padding,
                                      activation_fn=None)
        # num_outputs not known -> Flatten, then set self.num_outputs
        # to the resulting number of nodes.
        else:
            self.last_layer_is_flattened = True
            layers.append(nn.Flatten())
            self.num_outputs = out_channels

        self._convs = nn.Sequential(*layers)

        # Build the value layers
        self._value_branch = SlimFC(out_channels,
                                    1,
                                    initializer=normc_initializer(0.01),
                                    activation_fn=None)

        # Holds the current "base" output (before logits layer).
        self._features = None

    @override(TorchModelV2)
    def forward(self, input_dict: Dict[str,
                                       TensorType], state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        self._features = input_dict["obs"].float().permute(0, 2, 1)
        conv_out = self._convs(self._features)
        self._features = conv_out

        if not self.last_layer_is_flattened:
            if self._logits:
                conv_out = self._logits(conv_out)
            if conv_out.shape[2] != 1:
                raise ValueError(
                    "Given `conv_filters` ({}) do not result in a [B, {} "
                    "(`num_outputs`), 1] shape (but in {})! Please adjust "
                    "your Conv1D stack such that the last dim is "
                    "1.".format(self.model_config["conv_filters"],
                                self.num_outputs, list(conv_out.shape)))
            logits = conv_out.squeeze(2)
            # logits = logits.squeeze(1)

            return logits, state
        else:
            return conv_out, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        if not self.last_layer_is_flattened:
            features = self._features.squeeze(2)
            # features = features.squeeze(2)
        else:
            features = self._features
        return self._value_branch(features).squeeze(1)

    def _hidden_layers(self, obs: TensorType) -> TensorType:
        res = self._convs(obs.permute(0, 2, 1))  # switch to channel-major
        # res = res.squeeze(3)
        res = res.squeeze(2)
        return res


class CustomCNN(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        self.activ = 'ReLU'

        self.custom_layers = []
        self.obs_shape = obs_space
        self.num_outputs = num_outputs

        self.feature_extractor = None
        self.final_layer = None
        self.create_layers()

    def create_layers(self):
        activ = getattr(nn, self.activ)
        activ_dense = getattr(nn, self.activ)

        branches = {}
        for obs_name, space in self.obs_shape.original_space.spaces.items():
            layers = []
            width, channel = space.shape

            layers.append(nn.Conv1d(channel, 64, kernel_size=5, stride=3))
            layers.append(activ())
            # layer_1 = activ(
            #     conv_1d(obs,
            #             'c1_' + str(index_observation),
            #             n_filters=64,
            #             filter_size=5,
            #             stride=3,
            #             init_scale=numpy.sqrt(2)))

            # layer_1 = tf.nn.dropout( layer_1, rate = 0.2)
            # layer_1 = tf.layers.batch_normalization(layer_1)

            layers.append(nn.Conv1d(64, 64, kernel_size=3, stride=2))
            layers.append(activ())
            # layer_2 = activ(
            #     conv_1d(layer_1,
            #             'c2_' + str(index_observation),
            #             n_filters=64,
            #             filter_size=3,
            #             stride=2,
            #             init_scale=numpy.sqrt(2)))

            # layer_2 = tf.nn.dropout(layer_2, rate = 0.2)
            # layer_2 = tf.layers.batch_normalization(layer_2)

            # layer_3 = activ(
            #     conv_1d(layer_2, 'c3_' + str(index_observation), n_filters=64, filter_size=3, stride=1,
            #             init_scale=numpy.sqrt(2)))

            # layer_3 = tf.nn.dropout(layer_3, rate = 0.3)
            # layer_3 = tf.layers.batch_normalization(layer_3)

            layers.append(nn.Flatten(1))
            # layer_3 = conv_to_fc(layer_2)

            layers.append(nn.Linear(576, 128))
            layers.append(activ_dense())
            # dense_1 = activ_dense(
            #     linear(layer_3,
            #            'fc1_' + str(index_observation),
            #            n_hidden=128,
            #            init_scale=numpy.sqrt(2)))
            # dense_1 = tf.nn.dropout(dense_1, rate = 0.2)
            # dense_1 = tf.layers.batch_normalization(dense_1)

            # dense_2 = activ(linear(dense_1, 'fc2_'+ str(index_observation), n_hidden=128, init_scale=numpy.sqrt(2)))
            # dense_2 = tf.nn.dropout(dense_2, rate = 0.3)
            # dense_2 = tf.layers.batch_normalization(dense_2)

            # features.append(dense_1)
            branches[obs_name] = nn.Sequential(*layers)
        self.feature_extractor = nn.ModuleDict(branches)

        # h_concat = tf.concat(features, 1)

        self._logits = nn.Sequential(
            nn.Linear(128 * len(branches), self.num_outputs), activ_dense())
        self._value_branch = nn.Sequential(nn.Linear(128 * len(branches), 1),
                                           activ_dense())
        # h_out_1 = activ_dense(
        #     linear(h_concat, 'dense_1', n_hidden=128,
        #            init_scale=numpy.sqrt(2)))
        # h_out_1 = tf.nn.dropout(h_out_1, rate = 0.2)
        # h_out_1 = tf.layers.batch_normalization(h_out_1)
        #
        # h_out_2 = activ(linear(h_out_1, 'dense_2', n_hidden=128, init_scale=numpy.sqrt(2)))
        # h_out_2 = tf.nn.dropout(h_out_2, rate = 0.2)
        # h_out_2 = tf.layers.batch_normalization(h_out_2)

    @override(TorchModelV2)
    def forward(self, input_dict: Dict[str,
                                       TensorType], state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        features = []
        for obs_name, obs in input_dict['obs'].items():
            features.append(self.feature_extractor[obs_name](obs.permute(
                0, 2, 1)))
        self._features = torch.cat(features, dim=1)
        logits = self._logits(self._features)

        return logits, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        return self._value_branch(self._features).squeeze(1)


ModelCatalog.register_custom_model("custom-fc", CustomFC)
ModelCatalog.register_custom_model("custom-cnn", CustomCNN)
ModelCatalog.register_custom_model("vision-1d", VisionNetwork1D)

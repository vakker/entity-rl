from typing import Any, Dict, List, Tuple, Union

import gym
import numpy as np
import torch
from ray.rllib.models import ModelCatalog
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

    pad_along_width = int(((out_width - 1) * stride_width + filter_width - in_width))
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

# def conv_1d(input_tensor,
#             scope,
#             *,
#             n_filters,
#             filter_size,
#             stride,
#             pad='VALID',
#             init_scale=1.0,
#             data_format='NHWC',
#             one_dim_bias=False):
#     """
#     Creates a 2d convolutional layer for TensorFlow
#     :param input_tensor: (TensorFlow Tensor) The input tensor for the convolution
#     :param scope: (str) The TensorFlow variable scope
#     :param n_filters: (int) The number of filters
#     :param filter_size:  (Union[int, [int], tuple<int, int>]) The filter size for the squared kernel matrix,
#     or the height and width of kernel filter if the input is a list or tuple
#     :param stride: (int) The stride of the convolution
#     :param pad: (str) The padding type ('VALID' or 'SAME')
#     :param init_scale: (int) The initialization scale
#     :param data_format: (str) The data format for the convolution weights
#     :param one_dim_bias: (bool) If the bias should be one dimentional or not
#     :return: (TensorFlow Tensor) 2d convolutional layer
#     """

#     channel_ax = 2
#     strides = [1, stride, 1]
#     bshape = [1, n_filters]

#     filter_width = filter_size

#     bias_var_shape = [n_filters] if one_dim_bias else [1, n_filters, 1]
#     n_input = input_tensor.get_shape()[channel_ax].value
#     wshape = [filter_width, n_input, n_filters]
#     with tf.variable_scope(scope):
#         weight = tf.get_variable("w", wshape, initializer=ortho_init_1d(init_scale))
#         bias = tf.get_variable("b",
#                                bias_var_shape,
#                                initializer=tf.constant_initializer(0.0))
#         bias = tf.reshape(bias, bshape)

#         # return bias + tf.nn.conv1d(input_tensor, weight, strides=strides, padding=pad, data_format='NWC')
#         return bias + tf.nn.conv1d(input_tensor, weight, stride=stride, padding=pad)


class CustomFC(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config,
                              name)
        nn.Module.__init__(self)

        self.torch_sub_model = TorchFC(obs_space, action_space, num_outputs, model_config,
                                       name)

    def forward(self, input_dict, state, seq_lens):
        input_dict["obs"] = input_dict["obs"].float()
        fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
        return fc_out, []

    def value_function(self):
        return torch.reshape(self.torch_sub_model.value_function(), [-1])


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
    def __init__(self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space,
                 num_outputs: int, model_config: ModelConfigDict, name: str):
        if not model_config.get("conv_filters"):
            model_config["conv_filters"] = get_filter_config(obs_space.shape)
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config,
                              name)
        nn.Module.__init__(self)

        activation = self.model_config.get("conv_activation")
        filters = self.model_config["conv_filters"]
        assert len(filters) > 0,\
            "Must provide at least 1 entry in `conv_filters`!"
        no_final_linear = self.model_config.get("no_final_linear")
        vf_share_layers = self.model_config.get("vf_share_layers")

        # Whether the last layer is the output of a Flattened (rather than
        # a n x (1,1) Conv2D).
        self.last_layer_is_flattened = False
        self._logits = None

        layers = []
        (h, w, in_channels) = obs_space.shape
        assert h == 1
        in_size = w
        for out_channels, kernel, stride in filters[:-1]:
            padding, out_size = same_padding_1d(in_size, kernel, stride)
            layers.append(
                SlimConv1d(in_channels,
                           out_channels,
                           kernel,
                           stride,
                           padding,
                           activation_fn=activation))
            in_channels = out_channels
            in_size = out_size

        out_channels, kernel, stride = filters[-1]

        # No final linear: Last layer is a Conv2D and uses num_outputs.
        if no_final_linear and num_outputs:
            layers.append(
                SlimConv1d(
                    in_channels,
                    num_outputs,
                    kernel,
                    stride,
                    None,  # padding=valid
                    activation_fn=activation))
            out_channels = num_outputs
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
                    activation_fn=activation))

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
        self._value_branch_separate = self._value_branch = None
        if vf_share_layers:
            self._value_branch = SlimFC(out_channels,
                                        1,
                                        initializer=normc_initializer(0.01),
                                        activation_fn=None)
        else:
            vf_layers = []
            (h, w, in_channels) = obs_space.shape
            assert h == 1
            in_size = w
            for out_channels, kernel, stride in filters[:-1]:
                padding, out_size = same_padding_1d(in_size, kernel, stride)
                vf_layers.append(
                    SlimConv1d(in_channels,
                               out_channels,
                               kernel,
                               stride,
                               padding,
                               activation_fn=activation))
                in_channels = out_channels
                in_size = out_size

            out_channels, kernel, stride = filters[-1]
            vf_layers.append(
                SlimConv1d(in_channels,
                           out_channels,
                           kernel,
                           stride,
                           None,
                           activation_fn=activation))

            vf_layers.append(
                SlimConv1d(in_channels=out_channels,
                           out_channels=1,
                           kernel=1,
                           stride=1,
                           padding=None,
                           activation_fn=None))
            self._value_branch_separate = nn.Sequential(*vf_layers)

        # Holds the current "base" output (before logits layer).
        self._features = None

    @override(TorchModelV2)
    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        self._features = input_dict["obs"].float().permute(0, 3, 1, 2)
        conv_out = self._convs(self._features)
        # Store features to save forward pass when getting value_function out.
        if not self._value_branch_separate:
            self._features = conv_out

        if not self.last_layer_is_flattened:
            if self._logits:
                conv_out = self._logits(conv_out)
            if conv_out.shape[2] != 1:
                raise ValueError("Given `conv_filters` ({}) do not result in a [B, {} "
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
        if self._value_branch_separate:
            value = self._value_branch_separate(self._features)
            value = value.squeeze(2)
            value = value.squeeze(1)
            return value
        else:
            if not self.last_layer_is_flattened:
                features = self._features.squeeze(2)
                # features = features.squeeze(2)
            else:
                features = self._features
            return self._value_branch(features).squeeze(1)

    def _hidden_layers(self, obs: TensorType) -> TensorType:
        res = self._convs(obs.permute(0, 3, 1, 2))  # switch to channel-major
        # res = res.squeeze(3)
        res = res.squeeze(2)
        return res


# class CustomCNN(TorchModelV2, nn.Module):
#     def __init__(self, obs_space, action_space, num_outputs, model_config, name):
#         TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config,
#                               name)
#         nn.Module.__init__(self)

#         self.activ = 'ReLU'

#         self.custom_layers = []
#         self.observation_shape = obs_space

#     def feature_extractor(self, input_observations, **kwargs):

#         activ = getattr(nn, self.activ)
#         activ_dense = getattr(nn, self.activ)

#         current_height = 0

#         features = []

#         # if input_observations.shape[0] != 128:
#         #     print(input_observations.shape)

#         for index_observation, shape in enumerate(self.observation_shape):

#             height, width, channel = shape

#             # Get observation

#             obs = input_observations[:, current_height:current_height +
#                                      height, :width, :channel]

#             if height == 1:
#                 obs = tf.squeeze(obs, axis=[1])
#             current_height += height

#             layer_1 = activ(
#                 conv_1d(obs,
#                         'c1_' + str(index_observation),
#                         n_filters=64,
#                         filter_size=5,
#                         stride=3,
#                         init_scale=numpy.sqrt(2)))

#             # layer_1 = tf.nn.dropout( layer_1, rate = 0.2)
#             # layer_1 = tf.layers.batch_normalization(layer_1)

#             layer_2 = activ(
#                 conv_1d(layer_1,
#                         'c2_' + str(index_observation),
#                         n_filters=64,
#                         filter_size=3,
#                         stride=2,
#                         init_scale=numpy.sqrt(2)))

#             # layer_2 = tf.nn.dropout(layer_2, rate = 0.2)
#             # layer_2 = tf.layers.batch_normalization(layer_2)

#             # layer_3 = activ(
#             #     conv_1d(layer_2, 'c3_' + str(index_observation), n_filters=64, filter_size=3, stride=1,
#             #             init_scale=numpy.sqrt(2)))

#             # layer_3 = tf.nn.dropout(layer_3, rate = 0.3)
#             # layer_3 = tf.layers.batch_normalization(layer_3)

#             layer_3 = conv_to_fc(layer_2)

#             dense_1 = activ_dense(
#                 linear(layer_3,
#                        'fc1_' + str(index_observation),
#                        n_hidden=128,
#                        init_scale=numpy.sqrt(2)))
#             # dense_1 = tf.nn.dropout(dense_1, rate = 0.2)
#             # dense_1 = tf.layers.batch_normalization(dense_1)

#             # dense_2 = activ(linear(dense_1, 'fc2_'+ str(index_observation), n_hidden=128, init_scale=numpy.sqrt(2)))
#             # dense_2 = tf.nn.dropout(dense_2, rate = 0.3)
#             # dense_2 = tf.layers.batch_normalization(dense_2)

#             features.append(dense_1)

#         h_concat = tf.concat(features, 1)

#         h_out_1 = activ_dense(
#             linear(h_concat, 'dense_1', n_hidden=128, init_scale=numpy.sqrt(2)))
#         # h_out_1 = tf.nn.dropout(h_out_1, rate = 0.2)
#         # h_out_1 = tf.layers.batch_normalization(h_out_1)
#         #
#         # h_out_2 = activ(linear(h_out_1, 'dense_2', n_hidden=128, init_scale=numpy.sqrt(2)))
#         # h_out_2 = tf.nn.dropout(h_out_2, rate = 0.2)
#         # h_out_2 = tf.layers.batch_normalization(h_out_2)

#         return h_out_1

# import os
# from typing import Any, Dict, List, Optional, Union

# from stable_baselines.common.callbacks import BaseCallback, EventCallback
# from stable_baselines.common.evaluation import evaluate_policy

# class CustomEvalCallBack(EventCallback):
#     """
#     Callback for evaluating an agent.
#     :param eval_env: (Union[gym.Env, VecEnv]) The environment used for initialization
#     :param callback_on_new_best: (Optional[BaseCallback]) Callback to trigger
#         when there is a new best model according to the `mean_reward`
#     :param n_eval_episodes: (int) The number of episodes to test the agent
#     :param eval_freq: (int) Evaluate the agent every eval_freq call of the callback.
#     :param log_path: (str) Path to a folder where the evaluations (`evaluations.npz`)
#         will be saved. It will be updated at each evaluation.
#     :param best_model_save_path: (str) Path to a folder where the best model
#         according to performance on the eval env will be saved.
#     :param deterministic: (bool) Whether the evaluation should
#         use a stochastic or deterministic actions.
#     :param render: (bool) Whether to render or not the environment during evaluation
#     :param verbose: (int)
#     """
#     def __init__(self,
#                  eval_env,
#                  n_eval_episodes: int = 5,
#                  eval_freq: int = 10000,
#                  log_path: str = None,
#                  deterministic: bool = True,
#                  callback_on_new_best: Optional[BaseCallback] = None,
#                  render: bool = False,
#                  verbose: int = 1):
#         super(CustomEvalCallBack, self).__init__(callback_on_new_best, verbose=verbose)
#         self.n_eval_episodes = n_eval_episodes
#         self.eval_freq = eval_freq
#         self.best_mean_reward = -numpy.inf
#         self.last_mean_reward = -numpy.inf
#         self.deterministic = deterministic
#         self.render = render

#         self.eval_env = eval_env
#         # Logs will be written in `evaluations.npz`
#         if log_path is not None:
#             log_path = os.path.join(log_path, 'evaluations')
#         self.log_path = log_path
#         self.evaluations_results = []
#         self.evaluations_timesteps = []
#         self.evaluations_length = []

#     def _init_callback(self):
#         # Does not work in some corner cases, where the wrapper is not the same

#         if self.log_path is not None:
#             os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

#     def _on_step(self) -> bool:

#         if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
#             # Sync training and eval env if there is VecNormalize

#             episode_rewards, episode_lengths = evaluate_policy(
#                 self.model,
#                 self.eval_env,
#                 n_eval_episodes=self.n_eval_episodes,
#                 render=self.render,
#                 deterministic=self.deterministic,
#                 return_episode_rewards=True)

#             mean_reward, std_reward = numpy.mean(episode_rewards), numpy.std(
#                 episode_rewards)
#             mean_ep_length, std_ep_length = numpy.mean(episode_lengths), numpy.std(
#                 episode_lengths)

#             print(self.num_timesteps, mean_reward)
#             # Keep track of the last evaluation, useful for classes that derive from this callback
#             self.last_mean_reward = mean_reward

#             if self.verbose > 0:
#                 print("Eval num_timesteps={}, "
#                       "episode_reward={:.2f} +/- {:.2f}".format(
#                           self.num_timesteps, mean_reward, std_reward))
#                 print("Episode length: {:.2f} +/- {:.2f}".format(
#                     mean_ep_length, std_ep_length))

#             if mean_reward > self.best_mean_reward:
#                 if self.verbose > 0:
#                     print("New best mean reward!")
#                 self.best_mean_reward = mean_reward
#                 # Trigger callback if needed
#                 if self.callback is not None:
#                     return self._on_event()

#         return True

ModelCatalog.register_custom_model("custom-fc", CustomFC)
ModelCatalog.register_custom_model("vision-1d", VisionNetwork1D)

import gym
from gym import spaces
from simple_playgrounds.utils import ActionTypes, SensorModality
from simple_playgrounds import Engine
import numpy

import tensorflow as tf
from stable_baselines.common.policies import LstmPolicy, CnnLstmPolicy, CnnLnLstmPolicy, CnnPolicy, RecurrentActorCriticPolicy


# from stable_baselines.sac.policies import LnCnnPolicy
from stable_baselines.ddpg.policies import LnCnnPolicy
from stable_baselines.common.input import observation_input
from stable_baselines.common.tf_layers import linear, conv_to_fc, ortho_init

def ortho_init_1d(scale=1.0):
    """
    Orthogonal initialization for the policy weights
    :param scale: (float) Scaling factor for the weights.
    :return: (function) an initialization function for the weights
    """

    # _ortho_init(shape, dtype, partition_info=None)
    def _ortho_init(shape, *_, **_kwargs):
        """Intialize weights as Orthogonal matrix.
        Orthogonal matrix initialization [1]_. For n-dimensional shapes where
        n > 2, the n-1 trailing axes are flattened. For convolutional layers, this
        corresponds to the fan-in, so this makes the initialization usable for
        both dense and convolutional layers.
        References
        ----------
        .. [1] Saxe, Andrew M., James L. McClelland, and Surya Ganguli.
               "Exact solutions to the nonlinear dynamics of learning in deep
               linear
        """
        # lasagne ortho init for tf
        shape = tuple(shape)
        flat_shape = (numpy.prod(shape[:-1]), shape[-1])

        gaussian_noise = numpy.random.normal(0.0, 1.0, flat_shape)
        u, _, v = numpy.linalg.svd(gaussian_noise, full_matrices=False)
        weights = u if u.shape == flat_shape else v  # pick the one with the correct shape
        weights = weights.reshape(shape)
        return (scale * weights[:shape[0], :shape[1]]).astype(numpy.float32)

    return _ortho_init

def conv_1d(input_tensor, scope, *, n_filters, filter_size, stride,
            pad='VALID', init_scale=1.0, data_format='NHWC', one_dim_bias=False):
    """
    Creates a 2d convolutional layer for TensorFlow
    :param input_tensor: (TensorFlow Tensor) The input tensor for the convolution
    :param scope: (str) The TensorFlow variable scope
    :param n_filters: (int) The number of filters
    :param filter_size:  (Union[int, [int], tuple<int, int>]) The filter size for the squared kernel matrix,
    or the height and width of kernel filter if the input is a list or tuple
    :param stride: (int) The stride of the convolution
    :param pad: (str) The padding type ('VALID' or 'SAME')
    :param init_scale: (int) The initialization scale
    :param data_format: (str) The data format for the convolution weights
    :param one_dim_bias: (bool) If the bias should be one dimentional or not
    :return: (TensorFlow Tensor) 2d convolutional layer
    """

    channel_ax = 2
    strides = [1, stride, 1]
    bshape = [1, n_filters]

    filter_width = filter_size

    bias_var_shape = [n_filters] if one_dim_bias else [1, n_filters, 1]
    n_input = input_tensor.get_shape()[channel_ax].value
    wshape = [filter_width, n_input, n_filters]
    with tf.variable_scope(scope):
        weight = tf.get_variable("w", wshape, initializer=ortho_init_1d(init_scale))
        bias = tf.get_variable("b", bias_var_shape, initializer=tf.constant_initializer(0.0))
        bias = tf.reshape(bias, bshape)

        # return bias + tf.nn.conv1d(input_tensor, weight, strides=strides, padding=pad, data_format='NWC')
        return bias + tf.nn.conv1d(input_tensor, weight, stride=stride, padding=pad)



class CustomPolicy(CnnLstmPolicy):


    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):

        activ = 'relu'

        self.custom_layers = []
        self.observation_shape = _kwargs['observation_shape']

        self.reuse = reuse

        super(CustomPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=False,
                                           n_lstm=128, cnn_extractor=self.feature_extractor, feature_extractiom='cnn',

                                            **_kwargs)



        # self._setup_init()



    def feature_extractor(self, input_observations, **kwargs):

        activ = tf.nn.relu

        current_height = 0

        features = []

        # if input_observations.shape[0] != 128:
        #     print(input_observations.shape)

        with tf.variable_scope("model", reuse=self.reuse):


            for index_observation, shape in enumerate(self.observation_shape):

                height, width, channel = shape

                # Get observation

                obs = input_observations[:, current_height: current_height + height, :width, :channel]

                if height == 1:
                    obs = tf.squeeze(obs, axis = [1])
                current_height+=height

                layer_1 = activ(
                    conv_1d(obs, 'c1_' + str(index_observation), n_filters=32, filter_size=7, stride=3,
                            init_scale=numpy.sqrt(2)))

                layer_2 = activ(
                    conv_1d(layer_1, 'c2_' + str(index_observation), n_filters=32, filter_size=5, stride=2,
                            init_scale=numpy.sqrt(2)))

                layer_3 = activ(
                    conv_1d(layer_2, 'c3_' + str(index_observation), n_filters=32, filter_size=3, stride=1,
                            init_scale=numpy.sqrt(2)))

                layer_3 = conv_to_fc(layer_3)

                dense_1 = activ(linear(layer_3, 'fc1_'+ str(index_observation), n_hidden=128, init_scale=numpy.sqrt(2)))

                dense_2 = activ(linear(dense_1, 'fc2_'+ str(index_observation), n_hidden=128, init_scale=numpy.sqrt(2)))

                features.append(dense_2)

            h_concat = tf.concat(features, 1)

            h_out_1 = activ(linear(h_concat, 'dense_1', n_hidden=128, init_scale=numpy.sqrt(2)))
            h_out_2 = activ(linear(h_out_1, 'dense_2', n_hidden=128, init_scale=numpy.sqrt(2)))

        return h_out_2


from stable_baselines.common.callbacks import EventCallback, BaseCallback
import os
from stable_baselines.common.evaluation import evaluate_policy
from typing import Union, List, Dict, Any, Optional


class CustomEvalCallBack(EventCallback):
    """
    Callback for evaluating an agent.
    :param eval_env: (Union[gym.Env, VecEnv]) The environment used for initialization
    :param callback_on_new_best: (Optional[BaseCallback]) Callback to trigger
        when there is a new best model according to the `mean_reward`
    :param n_eval_episodes: (int) The number of episodes to test the agent
    :param eval_freq: (int) Evaluate the agent every eval_freq call of the callback.
    :param log_path: (str) Path to a folder where the evaluations (`evaluations.npz`)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: (str) Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: (bool) Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: (bool) Whether to render or not the environment during evaluation
    :param verbose: (int)
    """
    def __init__(self, eval_env,
                 n_eval_episodes: int = 5,
                 eval_freq: int = 10000,
                 log_path: str = None,
                 deterministic: bool = True,
                 callback_on_new_best: Optional[BaseCallback] = None,
                 render: bool = False,
                 verbose: int = 1):
        super(CustomEvalCallBack, self).__init__(callback_on_new_best, verbose=verbose)
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -numpy.inf
        self.last_mean_reward = -numpy.inf
        self.deterministic = deterministic
        self.render = render

        self.eval_env = eval_env
        # Logs will be written in `evaluations.npz`
        if log_path is not None:
            log_path = os.path.join(log_path, 'evaluations')
        self.log_path = log_path
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_length = []

    def _init_callback(self):
        # Does not work in some corner cases, where the wrapper is not the same

        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def _on_step(self) -> bool:

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize

            episode_rewards, episode_lengths = evaluate_policy(self.model, self.eval_env,
                                                               n_eval_episodes=self.n_eval_episodes,
                                                               render=self.render,
                                                               deterministic=self.deterministic,
                                                               return_episode_rewards=True)

            mean_reward, std_reward = numpy.mean(episode_rewards), numpy.std(episode_rewards)
            mean_ep_length, std_ep_length = numpy.mean(episode_lengths), numpy.std(episode_lengths)

            print( self.num_timesteps, mean_reward)
            # Keep track of the last evaluation, useful for classes that derive from this callback
            self.last_mean_reward = mean_reward

            if self.verbose > 0:
                print("Eval num_timesteps={}, "
                      "episode_reward={:.2f} +/- {:.2f}".format(self.num_timesteps, mean_reward, std_reward))
                print("Episode length: {:.2f} +/- {:.2f}".format(mean_ep_length, std_ep_length))

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                self.best_mean_reward = mean_reward
                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()

        return True


def test(scaled_images, **kwargs):

    activ = tf.nn.relu

    obs_shape = [ [1,64,3], [1,64,1], [1,64,1]]
    current_height = 0

    features = []

    for index_observation, shape in enumerate(obs_shape):

        height, width, channel = shape

        # Get observation

        obs = scaled_images[:, current_height: current_height + height, :width, :channel]

        if height == 1:
            obs = tf.squeeze(obs, axis = [1])
        current_height+=height

        layer_1 = activ(
            conv_1d(obs, 'c1_' + str(index_observation), n_filters=32, filter_size=7, stride=3,
                    init_scale=numpy.sqrt(2)))

        layer_2 = activ(
            conv_1d(layer_1, 'c2_' + str(index_observation), n_filters=32, filter_size=5, stride=2,
                    init_scale=numpy.sqrt(2)))

        layer_3 = activ(
            conv_1d(layer_2, 'c3_' + str(index_observation), n_filters=32, filter_size=3, stride=1,
                    init_scale=numpy.sqrt(2)))

        layer_3 = conv_to_fc(layer_3)

        dense_1 = activ(linear(layer_3, 'fc1_'+ str(index_observation), n_hidden=128, init_scale=numpy.sqrt(2)))

        dense_2 = activ(linear(dense_1, 'fc2_'+ str(index_observation), n_hidden=128, init_scale=numpy.sqrt(2)))

        features.append(dense_2)

    h_concat = tf.concat(features, 1)

    h_out_1 = activ(linear(h_concat, 'dense_1', n_hidden=128, init_scale=numpy.sqrt(2)))
    h_out_2 = activ(linear(h_out_1, 'dense_2', n_hidden=128, init_scale=numpy.sqrt(2)))

    return h_out_2
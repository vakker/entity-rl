import random
from abc import ABC, abstractmethod

import gym
import numpy as np
from ale_py import ALEInterface, LoggerMode
from ray.rllib.env.wrappers import atari_wrappers as wrappers
from skimage import transform
from skimage.util import img_as_ubyte

ALEInterface.setLoggerMode(LoggerMode.Error)


class SimpleCorridor(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.

    You can configure the length of the corridor via the env config."""

    def __init__(self, config):
        self.end_pos = 10
        self.cur_pos = 0
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            0.0, self.end_pos, shape=(1,), dtype=np.float32
        )
        # Set the seed. This is only used for the final (reach goal) reward.
        self.seed(0)

    def reset(self, *, seed=None, return_info=False, options=None):
        self.cur_pos = 0
        return [self.cur_pos]

    def step(self, action):
        assert action in [0, 1], action
        if action == 0 and self.cur_pos > 0:
            self.cur_pos -= 1
        elif action == 1:
            self.cur_pos += 1
        done = self.cur_pos >= self.end_pos
        # Produce a random reward when we reach the goal.
        return [self.cur_pos], random.random() * 2 if done else -0.1, done, {}

    def seed(self, seed=None):
        random.seed(seed)

    def render(self, mode="human"):
        return None


class AtariEnv(gym.Env, ABC):
    """Custom Environment that follows gym interface"""

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, config):
        seed = config.get("seed", 0)
        seed = (seed + id(self)) % (2**32)
        random.seed(seed)
        np.random.seed(seed)

        self._env = gym.make(config["pg_name"])
        if config.get("wrap", True):
            self._env = wrap_deepmind(self._env)

        self.video_dir = config.get("video_dir")

        self.episodes = 0

        self.time_steps = 0
        self.obs_raw = None
        self._crop = [0, 0]
        # self._crop = [25, 10]

        self.action_space = self._env.action_space
        self._obs_space = None
        self._config = config

    @property
    def observation_space(self):
        if self._obs_space is None:
            self._obs_space = self._set_obs_space()

        return self._obs_space

    def _set_obs_space(self):
        orig = self._env.observation_space
        cropped_shape = (orig.shape[0] - sum(self._crop), orig.shape[1], orig.shape[2])
        return gym.spaces.Box(0, 1, cropped_shape, dtype=np.float32)

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        return self._process_obs(obs), reward, done, info

    def reset(self, *, seed=None, return_info=False, options=None):
        assert not return_info

        obs = self._env.reset()
        self.episodes += 1

        return self._process_obs(obs)

    def render(self, mode="human"):
        if self.video_dir is None:
            return None

        raise NotImplementedError()

    def close(self):
        self._env.close()

    def crop_obs(self, obs):
        if self._crop[1] > 0:
            return obs[self._crop[0] : -self._crop[1]]

        return obs[self._crop[0] :]

    def _process_obs(self, obs):
        return self.process_obs(self.crop_obs(obs))

    @abstractmethod
    def process_obs(self, obs):
        pass

    @abstractmethod
    def full_scenario(self):
        pass


class AtariRaw(AtariEnv):
    def process_obs(self, obs):
        self.obs_raw = obs
        return obs / 255

    def full_scenario(self):
        return self.obs_raw


class ResizeEnv(gym.ObservationWrapper):
    def __init__(self, env, size):
        super().__init__(env)
        self._size = size

        self.observation_space = gym.spaces.Box(
            low=np.amin(env.observation_space.low),
            high=np.amax(env.observation_space.high),
            shape=(size[0], size[1], env.observation_space.shape[-1]),
            dtype=env.observation_space.dtype,
        )

    def observation(self, observation):
        return img_as_ubyte(
            transform.resize(
                observation,
                self._size,
                order=0,
                anti_aliasing=False,
            )
        )


class SkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break

        return obs, total_reward, done, info


def wrap_deepmind(env, framestack=True, skip=0, stack=4, resize=None):
    """Configure environment for DeepMind-style Atari. See:
    https://github.com/ray-project/ray/blob/master/rllib/env/wrappers/atari_wrappers.py

    For default env settings see:
    https://www.gymlibrary.dev/environments/atari/

    Note: ALE v5 envs have frameskip by default.
    """

    env = wrappers.MonitorEnv(env)
    env = wrappers.NoopResetEnv(env, noop_max=30)
    if skip > 0:
        env = SkipEnv(env, skip=skip)  # Don't use max, only skip
        # env = wrappers.MaxAndSkipEnv(env, skip=skip)

    env = wrappers.EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = wrappers.FireResetEnv(env)

    # env = wrappers.WarpFrame(env, dim) # don't warp
    # env = wrappers.ScaledFloatFrame(env)  # TODO: use for dqn?
    # env = ClipRewardEnv(env)  # reward clipping is handled by policy eval

    if framestack is True:
        env = wrappers.FrameStack(env, stack)

    if resize:
        env = ResizeEnv(env, resize)

    return env

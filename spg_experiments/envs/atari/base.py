import random
from abc import ABC, abstractmethod

import gym
import numpy as np


class AtariEnv(gym.Env, ABC):
    """Custom Environment that follows gym interface"""

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, config):
        seed = config.get("seed", 0)
        seed = (seed + id(self)) % (2 ** 32)
        random.seed(seed)
        np.random.seed(seed)

        self._env = gym.make(config["pg_name"])

        self.video_dir = config.get("video_dir")

        self.episodes = 0

        self.time_steps = 0
        self.obs_raw = None
        self._crop = [25, 10]

        self.action_space = self._env.action_space
        self._set_obs_space()

    @abstractmethod
    def _set_obs_space(self):
        pass

    def step(self, action):
        obs, reward, done, info = self._env.step(action)

        return self._process_obs(obs), reward, done, info

    def reset(self):
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
        return obs[self._crop[0] : -self._crop[1]]

    def _process_obs(self, obs):
        return self.process_obs(self.crop_obs(obs))

    @abstractmethod
    def process_obs(self, obs):
        pass

    @abstractmethod
    def full_scenario(self):
        pass


class AtariRaw(AtariEnv):
    def _set_obs_space(self):
        orig = self._env.observation_space
        cropped_shape = (orig.shape[0] - sum(self._crop), orig.shape[1], orig.shape[2])
        self.observation_space = gym.spaces.Box(0, 1, cropped_shape, dtype=np.float32)

    def process_obs(self, obs):
        self.obs_raw = obs
        return obs / 255

    def full_scenario(self):
        return self.obs_raw

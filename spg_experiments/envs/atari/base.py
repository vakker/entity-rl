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

        self._env = gym.make(config["game_name"])

        self.video_dir = config.get("video_dir")

        self.time_limit = 1000  # TODO: ?
        self.episodes = 0

        self.action_space = self._env.action_space
        self._set_obs_space()

        self.time_steps = 0
        self.obs_raw = None

    @abstractmethod
    def _set_obs_space(self):
        pass

    def step(self, action):
        obs, reward, done, info = self._env(action)
        self.time_steps += 1
        done = done or self.time_steps >= self.time_limit

        return self.process_obs(obs), reward, done, info

    def full_scenario(self):
        return self.obs_raw

    def reset(self):
        obs = self._env.reset()
        self.episodes += 1
        self.time_steps = 0

        return self.process_obs(obs)

    def render(self, mode="human"):
        if self.video_dir is None:
            return None

        raise NotImplementedError()

    def close(self):
        self._env.close()

    def process_obs(self, obs):
        self.obs_raw = obs
        return self._process_obs(obs)

    @abstractmethod
    def _process_obs(self, obs):
        pass


class AtariRaw(AtariEnv):
    def _set_obs_space(self):
        self.observation_space = self._env.observation_space

    def _process_obs(self, obs):
        return obs

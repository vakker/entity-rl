import warnings
from collections import deque

import gymnasium as gym
import numpy as np
from ray.rllib.utils.spaces.repeated import Repeated

from entity_rl.envs.atari.base import SkipEnv

from .base import PlaygroundEnv, type_str


class PgSet(PlaygroundEnv):
    max_elements = 80

    def _create_agent(self, agent_type, sensors_name, keyboard=False):
        if sensors_name != "semantic":
            warnings.warn("Setting sensors_name to semantic.")

            sensors_name = "semantic"

        super()._create_agent(agent_type, sensors_name, keyboard)

    @property
    def x_shape(self):
        return (3 + len(self.entity_types_map),)

    @property
    def entity_features(self):
        return {
            "x": Repeated(
                gym.spaces.Box(-1, 1, shape=self.x_shape, dtype=np.float32),
                self.max_elements,
            ),
        }

    def _set_obs_space(self):
        self.observation_space = gym.spaces.Dict(self.entity_features)

    def create_entity_features(self, obs):
        x = []

        # Agent "background" info
        if type_str(self.agent) in self.entity_types_map:
            ent_type = np.zeros((len(self.entity_types_map),), dtype=np.float32)
            ent_type[self.entity_types_map[type_str(self.agent)]] = 1

            location = np.array([0, np.cos(self.agent.angle), np.sin(self.agent.angle)])
            node_feat = np.concatenate([location, ent_type]).astype(np.float32)
            x.append(node_feat)

        for detection in obs["semantic"]:
            location = np.array(
                [
                    detection.distance,
                    np.cos(detection.angle),
                    np.sin(detection.angle),
                ],
            )
            ent_type = np.zeros((len(self.entity_types_map),), dtype=np.float32)
            ent_type[self.entity_types_map[type_str(detection.entity)]] = 1

            node_feat = np.concatenate([location, ent_type]).astype(np.float32)
            x.append(node_feat)

        return x

    def process_obs(self, obs):
        x = self.create_entity_features(obs)
        sensor_values = {"x": x}
        return sensor_values


class FrameStackSet(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames."""
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)

        # +1 because of the frame depth
        x_shape = (env.x_shape[0] + 1,)
        entity_features = {
            "x": Repeated(
                gym.spaces.Box(-1, 1, shape=x_shape, dtype=np.float32),
                env.max_elements * k,
            ),
        }
        self.observation_space = gym.spaces.Dict(entity_features)

    def reset(self, *, seed=None, options=None):
        ob, infos = self.env.reset(seed=seed, options=options)
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob(), infos

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, terminated, truncated, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        outputs = []
        for i, frame in enumerate(self.frames):
            for x in frame["x"]:
                outputs.append(np.concatenate([x, [i / self.k]]).astype(np.float32))

        outputs = {"x": outputs}
        return outputs


def wrap_deepmind_spg(env, skip=0, stack=4):
    if skip > 0:
        env = SkipEnv(env, skip=skip)

    if stack > 1:
        env = FrameStackSet(env, stack)

    return env


class PgSetWrapped(gym.Env):
    def __init__(self, config):
        wrap = config.pop("wrap")
        env = PgSet(config)
        if wrap:
            env = wrap_deepmind_spg(env, **wrap)

        self._env = env

    def step(self, *args, **kwargs):
        return self._env.step(*args, **kwargs)

    def reset(self, *args, **kwargs):
        return self._env.reset(*args, **kwargs)

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

    def render(self):
        pass

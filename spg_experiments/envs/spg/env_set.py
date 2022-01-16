import warnings

import gym
import numpy as np
from ray.rllib.utils.spaces.repeated import Repeated

from .base import PlaygroundEnv, type_str


class PgSet(PlaygroundEnv):
    max_elements = 80

    def _create_agent(self, agent_type, sensors_name, fov, resolution, keyboard=False):
        if sensors_name != "semantic":
            warnings.warn("Setting sensors_name to semantic.")

            sensors_name = "semantic"

        super()._create_agent(agent_type, sensors_name, fov, resolution, keyboard)

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

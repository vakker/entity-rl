import warnings
from collections import OrderedDict

import gym
import numpy as np
from ray.rllib.utils.spaces.repeated import Repeated

from .base import PlaygroundEnv, type_str


class PgSet(PlaygroundEnv):
    def _create_agent(self, agent_type, sensors_name, fov, resolution):
        if sensors_name != "semantic":
            warnings.warn("Setting sensors_name to semantic.")

            sensors_name = "semantic"

        super()._create_agent(agent_type, sensors_name, fov, resolution)

    def _set_obs_space(self):
        type_shape = (len(self.entity_types_map),)
        elements_space = gym.spaces.Dict(
            {
                "location": gym.spaces.Box(-1, 1, shape=(3,)),
                "type": gym.spaces.Box(0, 1, shape=type_shape),
            }
        )
        max_elements = 100
        self.observation_space = Repeated(elements_space, max_len=max_elements)

    def process_obs(self, obs):
        sensor_values = []
        for detection in obs["semantic"]:
            location = np.array(
                [detection.distance, np.cos(detection.angle), np.sin(detection.angle)]
            )
            ent_type = np.zeros((len(self.entity_types_map),), dtype=np.float32)
            ent_type[self.entity_types_map[type_str(detection.entity)]] = 1
            sensor_values.append(
                OrderedDict([("location", location), ("type", ent_type)])
            )
        return sensor_values

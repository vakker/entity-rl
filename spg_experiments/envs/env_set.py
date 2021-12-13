from collections import OrderedDict

import gym
import numpy as np
from ray.rllib.utils.spaces.repeated import Repeated

from .base import PlaygroundEnv


class PlaygroundEnvSemantic(PlaygroundEnv):
    def _create_agent(self, agent_type, sensors_name):
        assert sensors_name == "semantic"
        super()._create_agent(agent_type, sensors_name)

    def _set_obs_space(self):
        type_shape = (len(self.playground.entity_types_map),)
        elements_space = gym.spaces.Dict(
            {
                "location": gym.spaces.Box(-1, 1, shape=(3,)),
                "type": gym.spaces.Box(0, 1, shape=type_shape),
            }
        )
        max_elements = 100
        self.observation_space = Repeated(elements_space, max_len=max_elements)

    @property
    def observations(self):
        sensor_values = []

        for detection in self.agent.sensors[0].sensor_values:
            location = np.array(
                [detection.distance, np.cos(detection.angle), np.sin(detection.angle)]
            )
            ent_type = np.zeros(
                (len(self.playground.entity_types_map),), dtype=np.float32
            )
            ent_type[self.playground.entity_types_map[type(detection.entity)]] = 1
            sensor_values.append(
                OrderedDict([("location", location), ("type", ent_type)])
            )
        return sensor_values

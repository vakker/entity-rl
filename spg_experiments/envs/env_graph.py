import warnings

import gym
import numpy as np
from ray.rllib.utils.spaces.repeated import Repeated

from .base import PlaygroundEnv, type_str


class PgGraph(PlaygroundEnv):
    def _create_agent(self, agent_type, sensors_name, fov, resolution, keyboard=False):
        if sensors_name != "semantic":
            warnings.warn("Setting sensors_name to semantic.")

            sensors_name = "semantic"

        super()._create_agent(agent_type, sensors_name, fov, resolution, keyboard)

    def _set_obs_space(self):
        x_shape = (3 + len(self.entity_types_map),)

        max_elements = 50
        self.observation_space = gym.spaces.Dict(
            {
                "x": Repeated(
                    gym.spaces.Box(-1, 1, shape=x_shape, dtype=np.float32),
                    max_elements,
                ),
                "edge_index": Repeated(
                    gym.spaces.Box(0, max_elements, shape=(2,), dtype=np.int64),
                    max_elements ** 2,
                ),
            }
        )

    def process_obs(self, obs):
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

        n_nodes = len(x)
        edge_index = [np.array([i, j]) for i in range(n_nodes) for j in range(n_nodes)]

        sensor_values = {"x": x, "edge_index": edge_index}
        return sensor_values

import gymnasium as gym
import numpy as np
from ray.rllib.utils.spaces.repeated import Repeated

from .env_set import PgSet


class PgGraph(PgSet):
    def _set_obs_space(self):
        d = {
            **self.entity_features,
            "edge_index": Repeated(
                gym.spaces.Box(0, self.max_elements, shape=(2,), dtype=np.int64),
                self.max_elements**2,
            ),
        }
        self.observation_space = gym.spaces.Dict(d)

    def process_obs(self, obs):
        x = self.create_entity_features(obs)
        sensor_values = {"x": x}

        n_nodes = len(x)
        edge_index = [np.array([i, j]) for i in range(n_nodes) for j in range(n_nodes)]

        sensor_values = {"x": x, "edge_index": edge_index}
        return sensor_values

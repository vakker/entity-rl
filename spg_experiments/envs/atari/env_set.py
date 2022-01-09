import gym
import numpy as np
from ray.rllib.utils.spaces.repeated import Repeated
from skimage.measure import label, regionprops

from .base import AtariEnv


class AtariSet(AtariEnv):
    # pylint: disable=no-self-use

    max_elements = 50

    @property
    def x_shape(self):
        # RGB, pos (row, col), size (row, col) -> 7
        return (7,)

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

        segments = get_segments(obs)
        props = regionprops(segments)

        for p in props:
            color = obs[p.coords[0][0], p.coords[0][1]] / 255
            pos = np.array(p.centroid) / obs.shape[:2]
            size = [
                p.bbox[2] - p.bbox[0],
                p.bbox[3] - p.bbox[1],
            ]
            size = np.array(size) / obs.shape[:2]

            node_feat = np.concatenate([color, pos, size]).astype(np.float32)
            x.append(node_feat)

        return x

    def _process_obs(self, obs):
        x = self.create_entity_features(obs)
        sensor_values = {"x": x}
        return sensor_values


def get_segments(obs):
    color_ids = get_color_ids(obs)
    return label(color_ids, background=0, connectivity=2)


def get_color_ids(obs):
    colors_np = obs.reshape((-1, 3))
    colors = list({tuple(c) for c in colors_np.tolist()})

    conds = {c: np.all(obs == c, axis=2) for c in colors}
    color_ids = np.zeros(obs.shape[:2], dtype=np.int)
    colors = sorted(colors, key=lambda c: np.sum(conds[c]), reverse=True)

    for i, c in enumerate(colors):
        cond = conds[c]
        color_ids[cond] = i

    return color_ids

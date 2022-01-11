import gym
import numpy as np
from ray.rllib.utils.spaces.repeated import Repeated
from skimage.color import label2rgb
from skimage.measure import label, regionprops
from skimage.util import img_as_ubyte

from .base import AtariEnv


class AtariSet(AtariEnv):
    # pylint: disable=no-self-use

    def __init__(self, config):
        if config["pg_name"] == "PongNoFrameskip-v4":
            self.max_elements = 5
        elif config["pg_name"] == "SkiingNoFrameskip-v4":
            self.max_elements = 30
        else:
            self.max_elements = 80

        self._color_cache = None
        self.segments = None

        super().__init__(config)

    def full_scenario(self):
        segm = label2rgb(self.segments)
        img = np.concatenate([self.obs_raw, img_as_ubyte(segm)], axis=1)
        return img

    def process_obs(self, obs):
        x = self.create_entity_features(obs)
        sensor_values = {"x": x}
        return sensor_values

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

        # TODO: optimize this further
        segments = self.get_segments(obs)
        self.obs_raw = obs
        self.segments = segments

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

    def get_segments(self, obs):
        colors_np = obs.reshape((-1, 3))
        if self._color_cache is None:
            self._color_cache = list({tuple(c) for c in colors_np.tolist()})

        color_ids = self._get_segments(colors_np)

        if np.any(color_ids == -1):
            self._color_cache = list({tuple(c) for c in colors_np.tolist()})
            color_ids = self._get_segments(colors_np)

        color_ids = color_ids.reshape(obs.shape[:2])
        return label(color_ids, background=0, connectivity=2)

    def _get_segments(self, colors_np):
        colors = self._color_cache

        conds = [np.all(colors_np == c, axis=1) for c in colors]
        sizes = np.array([np.sum(c) for c in conds])
        color_order = np.argsort(sizes)[::-1]
        color_ids = -1 * np.ones(colors_np.shape[0], dtype=np.int)

        for i, color_idx in enumerate(color_order):
            cond = conds[color_idx]
            color_ids[cond] = i

        return color_ids

from os import path as osp

import gymnasium as gym
import numpy as np
import yaml
from ray.rllib.utils.spaces.repeated import Repeated
from skimage.color import label2rgb
from skimage.draw import disk, rectangle, rectangle_perimeter
from skimage.measure import label, regionprops
from skimage.util import img_as_ubyte

from .base import AtariEnv


class AtariSet(AtariEnv):
    def __init__(self, config):
        super().__init__(config)

        self._bg_color = None
        self.segments = None
        self.props = None

        self.stack_size = self._env.observation_space.shape[-1] // 3

        dir_path = osp.dirname(osp.abspath(__file__))
        with open(osp.join(dir_path, "max-x-for-game.yml")) as yaml_f:
            max_x = yaml.safe_load(yaml_f)

        if config["pg_name"] in max_x:
            max_elements = max_x[config["pg_name"]]
        else:
            max_elements = max(x for _, x in max_x.items())
            print("No info for ", config["pg_name"], "setting max x to ", max_elements)

        self.max_elements = int(max_elements * 1.5)
        self.max_elements *= self.stack_size

    # TODO: change this to render env or something
    def full_scenario(self):
        segm = self.obs_raw.copy()

        for p in self.props:
            size = [
                p.bbox[2] - p.bbox[0],
                p.bbox[3] - p.bbox[1],
            ]

            rr, cc = rectangle(start=p.bbox[:2], extent=size, shape=segm.shape)

            color = self.obs_raw[p.coords[0][0], p.coords[0][1]]
            segm[rr, cc] = color
            rr, cc = rectangle_perimeter(
                start=p.bbox[:2], extent=size, shape=segm.shape
            )
            segm[rr, cc] = [0, 0, 255]

            rr, cc = disk(p.centroid, 1, shape=segm.shape)
            segm[rr, cc] = [255, 0, 0]

        line = img_as_ubyte(np.ones((segm.shape[0], 10, 3)))
        img = np.concatenate(
            [
                img_as_ubyte(self.obs_raw),
                line,
                img_as_ubyte(segm),
                line,
                img_as_ubyte(label2rgb(self.segments)),
            ],
            axis=1,
        )
        return img

    def process_obs(self, obs):
        x = self.create_entity_features(obs)
        sensor_values = {"x": x}
        return sensor_values

    @property
    def x_shape(self):
        # RGB, pos (row, col), size (row, col), stack depth -> 8
        return (8,)

    @property
    def entity_features(self):
        return {
            "x": Repeated(
                gym.spaces.Box(-1, 1, shape=self.x_shape, dtype=np.float32),
                self.max_elements,
            ),
        }

    def _set_obs_space(self):
        return gym.spaces.Dict(self.entity_features)

    def create_entity_features(self, obs):
        x = []

        for stack_nr in range(self.stack_size):
            c_min = stack_nr * 3
            c_max = (stack_nr + 1) * 3
            frame = obs[:, :, c_min:c_max]

            segments = self.get_segments(frame)
            props = regionprops(segments)

            for p in props:
                color = frame[p.coords[0][0], p.coords[0][1]] / 255
                pos = np.array(p.centroid) / frame.shape[:2]
                size = [
                    p.bbox[2] - p.bbox[0],
                    p.bbox[3] - p.bbox[1],
                ]
                size = np.array(size) / frame.shape[:2]

                node_feat = np.concatenate(
                    [
                        color,
                        pos,
                        size,
                        [stack_nr / self.stack_size],
                    ]
                ).astype(np.float32)
                x.append(node_feat)

        self.obs_raw = frame
        self.segments = segments
        self.props = props

        if len(x) > self.max_elements:
            print(
                f"Num elements ({len(x)}) larger than max ({self.max_elements}) for ",
                self._config["pg_name"],
            )

        # assert len(x) <= self.max_elements, (
        #     f"Num elements ({len(x)}) larger than max",
        #     f"({self.max_elements}) for " + self._config["pg_name"],
        # )

        return x

    def get_segments(self, obs):
        scaler = np.array([1, 500, 500 * 500])
        colors = obs @ scaler

        if self._bg_color is None:
            counts = np.bincount(colors.reshape(-1))
            self._bg_color = np.argmax(counts)

        labels = label(colors, background=self._bg_color, connectivity=2)
        return labels

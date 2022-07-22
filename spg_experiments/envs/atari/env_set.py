import gym
import numpy as np
from ray.rllib.utils.spaces.repeated import Repeated
from skimage.color import label2rgb
from skimage.draw import disk, rectangle, rectangle_perimeter
from skimage.measure import label, regionprops
from skimage.util import img_as_ubyte

from .base import AtariEnv


class AtariSet(AtariEnv):
    # pylint: disable=no-self-use

    def __init__(self, config):
        super().__init__(config)

        self._color_cache = None
        self.segments = None
        self.props = None

        self.stack_size = self._env.observation_space.shape[-1] // 3

        if config["pg_name"] == "PongNoFrameskip-v4":
            self.max_elements = 10
        elif config["pg_name"] == "SkiingNoFrameskip-v4":
            self.max_elements = 30
        else:
            self.max_elements = 80

        self.max_elements *= self.stack_size

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

        assert (
            len(x) < self.max_elements
        ), f"Num elements ({len(x)}) larger than max ({self.max_elements})"

        return x

    def get_segments(self, obs):
        # TODO: optimize this further
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

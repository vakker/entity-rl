import sys
from abc import ABC, abstractmethod

import numpy as np
from gymnasium import spaces
from skimage.measure import label, regionprops
from torch import nn

module = sys.modules[__name__]


class EntityEncoder(nn.Module, ABC):
    def __init__(self, model_config, obs_space):
        super().__init__()

        self._config = model_config
        self._obs_space = obs_space

    @abstractmethod
    def get_out_channels(self):
        pass

    @abstractmethod
    def forward(self, inputs):
        pass


class EntityPassThrough(EntityEncoder):
    def __init__(self, model_config, obs_space):
        assert isinstance(obs_space, spaces.Graph)
        super().__init__(model_config, obs_space)

    def get_out_channels(self):
        # TODO: add global features
        out_channels = {
            "node_features": self._obs_space.node_space.shape,
            "edge_features": self._obs_space.edge_space.shape,
            "global_features": None,
        }
        return out_channels

    def forward(self, inputs):
        return inputs


class ManualEntityEncoder(EntityEncoder):
    def __init__(self, model_config, obs_space):
        assert isinstance(obs_space, spaces.Box)
        super().__init__(model_config, obs_space)

        # Due to frame stacking
        self.stack_size = self._obs_space.shape[-1] // 3

    def get_out_channels(self):
        # edge_index?
        out_channels = {
            "node_features": self.x_shape,
            "edge_features": self.e_shape,
            "global_features": None,
        }
        return out_channels

    def forward(self, inputs):
        x = self.create_entity_features(inputs)

        if not self._config["add_edges"]:
            return {"x": x}

        n_nodes = len(x)
        # NOTE: this is simple fully connected graph
        edge_index = [np.array([i, j]) for i in range(n_nodes) for j in range(n_nodes)]

        # NOTE: this doesn't have edge features, only the
        # connections
        return {"x": x, "edge_index": edge_index}

    @property
    def e_shape(self):
        if not self._config["add_edges"]:
            return None
        # Adjust for the number of channels
        return (1,)

    @property
    def x_shape(self):
        # colour (R, G, B), pos (row, col), size (row, col), stack depth -> 8
        return (8,)

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

        # self.obs_raw = frame
        # self.segments = segments
        # self.props = props

        # if len(x) > self.max_elements:
        #     print(
        #         f"Num elements ({len(x)}) larger than max ({self.max_elements}) for ",
        #         self._config["pg_name"],
        #     )

        # assert len(x) <= self.max_elements, (
        #     f"Num elements ({len(x)}) larger than max",
        #     f"({self.max_elements}) for " + self._config["pg_name"],
        # )

        return x

    def get_segments(self, obs):
        scaler = np.array([1, 500, 500 * 500])
        colors = obs @ scaler

        counts = np.bincount(colors.reshape(-1))
        bg_color = np.argmax(counts)

        labels = label(colors, background=bg_color, connectivity=2)
        return labels

    # TODO: this is used for rendering, needs to be integrated with the
    # policy
    # def full_scenario(self):
    #     segm = self.obs_raw.copy()

    #     for p in self.props:
    #         size = [
    #             p.bbox[2] - p.bbox[0],
    #             p.bbox[3] - p.bbox[1],
    #         ]

    #         rr, cc = rectangle(start=p.bbox[:2], extent=size, shape=segm.shape)

    #         color = self.obs_raw[p.coords[0][0], p.coords[0][1]]
    #         segm[rr, cc] = color
    #         rr, cc = rectangle_perimeter(
    #             start=p.bbox[:2], extent=size, shape=segm.shape
    #         )
    #         segm[rr, cc] = [0, 0, 255]

    #         rr, cc = disk(p.centroid, 1, shape=segm.shape)
    #         segm[rr, cc] = [255, 0, 0]

    #     line = img_as_ubyte(np.ones((segm.shape[0], 10, 3)))
    #     img = np.concatenate(
    #         [
    #             img_as_ubyte(self.obs_raw),
    #             line,
    #             img_as_ubyte(segm),
    #             line,
    #             img_as_ubyte(label2rgb(self.segments)),
    #         ],
    #         axis=1,
    #     )
    #     return img

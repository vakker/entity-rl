# pylint: disable=self-assigning-variable

import os
import random
from abc import ABC, abstractmethod
from os import path as osp

import gym
import numpy as np
from gym import spaces
from simple_playgrounds.agent import agents, controllers
from simple_playgrounds.device import sensors
from simple_playgrounds.element.elements.activable import Dispenser
from simple_playgrounds.engine import Engine
from simple_playgrounds.playground.playground import PlaygroundRegister
from simple_playgrounds.playground.playgrounds.rl import foraging
from skimage import io as skio

from spg_experiments import playgrounds

# Import needed because of the register, and this is needed because of the linters
foraging = foraging
playgrounds = playgrounds


class PlaygroundEnv(gym.Env, ABC):
    """Custom Environment that follows gym interface"""

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, config):
        sensors_name = config.get("sensors_name", "rgb_depth")
        sensors_fov = config.get("sensors_fov", 360)
        sensors_res = config.get("sensors_res", 64)
        multisteps = config.get("multisteps")
        keyboard = config.get("keyboard")
        self.include_agent_in_obs = config.get("include_agent", False)
        continuous_action_space = config.get("continuous_actions", False)

        seed = config.get("seed", 0)
        seed = (seed + id(self)) % (2 ** 32)
        random.seed(seed)
        np.random.seed(seed)

        self.video_dir = config.get("video_dir")

        pg_name = config["pg_name"].split("/")
        self.playground = PlaygroundRegister.playgrounds[pg_name[0]][pg_name[1]]()
        self.playground.time_limit = 1000
        self.time_limit = self.playground.time_limit
        self.episodes = 0
        self._entity_types = None

        self._create_agent("base", sensors_name, sensors_fov, sensors_res, keyboard)
        self._set_action_space(continuous_action_space)
        self._set_obs_space()

        self.multisteps = None
        if multisteps is not None:
            assert isinstance(multisteps, int)
            self.multisteps = multisteps
        self.time_steps = 0

    @property
    def entity_types_map(self):
        if self._entity_types is None:
            self._entity_types = {}
            element_types = []

            if self.include_agent_in_obs:
                element_types += [type_str(self.agent)]

            element_types += [type_str(e) for e in self.playground.elements]

            dispensers = [
                elem for elem in self.playground.elements if isinstance(elem, Dispenser)
            ]
            element_types += [d.elem_class_produced.__name__ for d in dispensers]
            element_types += [
                s.entity_produced.__name__ for s in self.playground.spawners
            ]

            entity_id = 0
            for element in element_types:
                if element not in self._entity_types:
                    self._entity_types[element] = entity_id
                    entity_id += 1

        return self._entity_types

    @abstractmethod
    def _set_obs_space(self):
        pass

    def _set_action_space(self, continuous_action_space):
        actuators = self.agent.controller.controlled_actuators
        self.continuous_action_space = continuous_action_space

        if self.continuous_action_space:
            lows = []
            highs = []

            for actuator in actuators:

                lows.append(actuator.min)
                highs.append(actuator.max)

            self.action_space = spaces.Box(
                low=np.array(lows).astype(np.float64),
                high=np.array(highs).astype(np.float64),
                dtype=np.float64,
            )

        else:
            act_spaces = []
            for actuator in actuators:
                act_spaces.append(3)

            self.action_space = spaces.MultiDiscrete(act_spaces)

    def _create_agent(self, agent_type, sensors_name, fov, resolution, keyboard=False):
        if agent_type == "base":
            agent_cls = agents.BaseAgent
        else:
            raise ValueError(f"Wrong agent_type: {agent_type}")

        if keyboard:
            cont = controllers.Keyboard()
        else:
            cont = controllers.External()

        agent = agent_cls(controller=cont)

        sensors_config = get_sensor_config(sensors_name, fov, resolution, 400)
        for sensor_cls, sensor_params in sensors_config:
            agent.add_sensor(
                sensor_cls(
                    anchor=agent.base_platform,
                    normalize=True,
                    invisible_elements=agent.parts,
                    **sensor_params,
                )
            )

        self.playground.add_agent(agent)

        self._engine = Engine(self.playground, full_surface=True)
        self.agent = agent
        assert self.agent in self._engine.agents

    @property
    def engine(self):
        return self._engine

    def get_current_timestep(self):
        return self.engine.elapsed_time

    def step(self, action):

        actions_to_game_engine = {}
        actions_dict = {}

        actuators = self.agent.controller.controlled_actuators
        for actuator, act in zip(actuators, action):
            if self.continuous_action_space:
                actions_dict[actuator] = act
            else:
                actions_dict[actuator] = [-1, 0, 1][act]

        actions_to_game_engine[self.agent] = actions_dict

        # Generate actions for other agents
        for agent in self.engine.agents:
            if agent is not self.agent:
                actions_to_game_engine[agent] = agent.controller.generate_actions()

        if self.multisteps is None:
            self.engine.step(actions_to_game_engine)
        else:
            self.engine.multiple_steps(actions_to_game_engine, self.multisteps)
        self.engine.update_observations()

        reward = self.agent.reward
        done = self.playground.done or not self.engine.game_on

        if hasattr(self.playground, "portals"):
            portal_used = int(any(p.energized for p in self.playground.portals))
        else:
            portal_used = 0

        info = {
            "data": {
                "running": {
                    "portal_used": portal_used,
                },
            },
        }

        if self.video_dir is not None:
            self.render()

        return self.observations, reward, done, info

    def full_scenario(self):
        return (255 * self.engine.generate_agent_image(self.agent)).astype(np.uint8)

    @property
    def observations(self):
        sensor_values = {}
        for sensor in self.agent.sensors:
            sensor_values[sensor.name] = sensor.sensor_values
        return self.process_obs(sensor_values)

    def reset(self):
        self.engine.reset()
        self.engine.elapsed_time = 0
        self.episodes += 1
        self.engine.update_observations()
        self.time_steps = 0

        return self.observations

    def render(self, mode="human"):
        if self.video_dir is None:
            return None

        img = self.full_scenario()

        step_id = self.engine.elapsed_time
        video_dir = osp.join(self.video_dir, str(id(self)), str(self.episodes))
        frame_path = osp.join(video_dir, f"f-{step_id:06d}.png")
        if not osp.exists(video_dir):
            os.makedirs(video_dir, exist_ok=True)

        skio.imsave(frame_path, img)

        if mode == "human":
            return img

        return None

    def close(self):
        self.engine.terminate()

    @abstractmethod
    def process_obs(self, obs):
        pass


class PgFlat(PlaygroundEnv):
    def process_obs(self, obs):
        obs_vec = []
        for _, v in obs.items():
            obs_vec.append(v.ravel())

        return np.concatenate(obs_vec)

    def _set_obs_space(self):
        elems = 0
        for sensor in self.agent.sensors:
            if isinstance(sensor.shape, int):
                elems += sensor.shape
            else:
                elems += np.prod(sensor.shape)

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(elems,),
            dtype=np.float64,
        )


class PgStacked(PlaygroundEnv):
    def process_obs(self, obs):
        obs_stacked = np.concatenate([v for k, v in obs.items()], axis=1)
        return {"stacked": obs_stacked}

    def _set_obs_space(self):
        channel_size = 0
        width = None
        for sensor in self.agent.sensors:
            if isinstance(sensor.shape, int):
                shape = (sensor.shape, 1)
            else:
                shape = sensor.shape

            assert len(shape) == 2

            if width is None:
                width = shape[0]
            else:
                assert width == shape[0], "Inconsistent obs width."

            channel_size += shape[1]

        box = spaces.Box(
            low=0,
            high=1,
            shape=(width, channel_size),
            dtype=np.float64,
        )
        self.observation_space = spaces.Dict({"stacked": box})


class PgDict(PlaygroundEnv):
    def process_obs(self, obs):
        return obs

    def _set_obs_space(self):
        d = {}
        for sensor in self.agent.sensors:
            if isinstance(sensor.shape, int):
                shape = (sensor.shape, 1)
            else:
                shape = sensor.shape

            d[sensor.name] = spaces.Box(
                low=0,
                high=1,
                shape=shape,
                dtype=np.float64,
            )

        self.observation_space = spaces.Dict(d)


def get_sensor_config(sensors_name, fov=360, resolution=64, max_range=300):
    if sensors_name == "blind":
        return [
            (
                sensors.BlindCamera,
                {
                    "fov": fov,
                    "resolution": resolution,
                    "name": "blind",
                },
            )
        ]

    if sensors_name == "semantic":
        return [
            (
                # FIXME: use SemanticRay instead
                sensors.PerfectSemantic,
                {
                    "max_range": max_range,
                    "resolution": resolution,
                    "name": "semantic",
                },
            )
        ]

    sensors_name = sensors_name.split("_")
    sensor_config = []
    for name in sensors_name:
        if name == "rgb":
            sensor_config.append(
                (
                    sensors.RgbCamera,
                    {
                        "fov": fov,
                        "max_range": max_range,
                        "resolution": resolution,
                        "name": "rgb",
                    },
                ),
            )

        if name in ["lidar", "depth"]:
            sensor_config.append(
                (
                    sensors.Lidar,
                    {
                        "fov": fov,
                        "max_range": max_range,
                        "resolution": resolution,
                        "name": "lidar",
                    },
                ),
            )

        if name == "touch":
            sensor_config.append(
                (
                    sensors.Touch,
                    {
                        "max_range": 2,
                        "fov": fov,
                        "resolution": resolution,
                        "name": "touch",
                    },
                ),
            )

    return sensor_config


def type_str(obj):
    return type(obj).__name__

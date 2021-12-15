import os
import random
from abc import ABC, abstractmethod
from os import path as osp

import gym
import numpy as np
from gym import spaces
from simple_playgrounds.agent import agents, controllers
from simple_playgrounds.device import sensors
from simple_playgrounds.engine import Engine
from simple_playgrounds.playground.playground import PlaygroundRegister
from simple_playgrounds.playground.playgrounds.rl import foraging
from skimage import io as skio

# Import needed because of the register, and this is needed because of the linters
_ = foraging


class PlaygroundEnv(gym.Env, ABC):
    """Custom Environment that follows gym interface"""

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, config):
        sensors_name = config["sensors_name"]
        multisteps = config.get("multisteps")
        continuous_action_space = True

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

        self._create_agent("base", sensors_name)
        self._set_action_space(continuous_action_space)
        self._set_obs_space()

        self.multisteps = None
        if multisteps is not None:
            assert isinstance(multisteps, int)
            self.multisteps = multisteps
        self.time_steps = 0

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
            raise NotImplementedError()

    def _create_agent(self, agent_type, sensors_name):
        if agent_type == "base":
            agent_cls = agents.BaseAgent
        else:
            raise ValueError(f"Wrong agent_type: {agent_type}")

        agent = agent_cls(controller=controllers.External())

        for sensor_cls, sensor_params in get_sensor_config(sensors_name):
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
            actuator.apply_action(act)

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
        # done = self.time_steps > 1000

        return (self.observations, reward, done, {})

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
        # TODO: verify this

        if self.video_dir is None:
            return None

        img = self.full_scenario()

        step_id = self.engine.elapsed_time
        video_dir = osp.join(self.video_dir, str(id(self)), str(self.episodes))
        frame_path = osp.join(video_dir, f"f-{step_id:03d}.png")
        if not osp.exists(video_dir):
            os.makedirs(video_dir, exist_ok=True)

        skio.imsave(frame_path, img)
        return img

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


class PgChannels(PlaygroundEnv):
    def process_obs(self, obs):
        obs_stacked = np.concatenate([v for k, v in obs.items()])
        return obs_stacked

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

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(width, channel_size),
            dtype=np.float64,
        )


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


def get_sensor_config(sensors_name):
    if sensors_name == "blind":
        return [(sensors.BlindCamera, {"resolution": 64})]

    if sensors_name == "semantic":
        return [(sensors.SemanticRay, {"range": 1000, "resolution": 1000})]

    sensors_name = sensors_name.split("_")
    sensor_config = []
    for name in sensors_name:
        if name == "rgb":
            sensor_config.append(
                (sensors.RgbCamera, {"fov": 180, "range": 300, "resolution": 64}),
            )

        if name == "depth":
            sensor_config.append(
                (sensors.Lidar, {"fov": 180, "range": 300, "resolution": 64}),
            )

        if name == "touch":
            sensor_config.append(
                (sensors.Touch, {"range": 2, "resolution": 64}),
            )

    return sensor_config

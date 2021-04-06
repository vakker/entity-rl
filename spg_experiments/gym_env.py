import os
import random
from os import path as osp

import cv2
import gym
import numpy as np
from gym import spaces
from simple_playgrounds import Engine
from simple_playgrounds.agents import agents, sensors
from simple_playgrounds.agents.parts import controllers
from simple_playgrounds.playground import PlaygroundRegister
from simple_playgrounds.utils.definitions import SensorTypes


class PlaygroundEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, config):
        super().__init__()

        playground_name = config['playground_name']
        agent_type = config['agent_type']
        sensors_name = config['sensors_name']
        seed = config.get('seed', 0)
        continuous_action_space = config.get('continuous_action_space', True)
        multisteps = config.get('multisteps')
        controller = config.get('controller', controllers.External())
        self.video_dir = config.get('video_dir')

        self.playground = PlaygroundRegister.playgrounds[playground_name[0]][
            playground_name[1]]()
        self.playground.time_limit = 1000
        self.episodes = 0

        seed = (seed + id(self)) % (2**32)
        random.seed(seed)
        np.random.seed(seed)

        if agent_type == 'base':
            agent_cls = agents.BaseAgent
        elif agent_type == 'arm':
            agent_cls = agents.FullAgent
        else:
            raise ValueError(f"Wrong agent_type: {agent_type}")

        agent = agent_cls(controller=controller)

        for sensor_name, sensor_params in get_sensor_params(sensors_name):
            if sensor_name == 'depth':
                sensor_cls = sensors.Lidar
                sensor_name = 'depth_0'

            elif sensor_name == 'rgb':
                sensor_cls = sensors.RgbCamera
                sensor_name = 'rgb_0'

            elif sensor_name == 'touch':
                sensor_cls = sensors.Touch
                sensor_name = 'touch_0'

            elif sensor_name == 'blind':
                sensor_cls = sensors.BlindCamera
                sensor_name = 'blind_0'

            else:
                raise NotImplementedError(
                    f'Sensor {sensor_name} not implemented')

            agent.add_sensor(
                sensor_cls(anchor=agent.base_platform,
                           normalize=True,
                           invisible_elements=agent.parts,
                           name=sensor_name,
                           **sensor_params))

        self.playground.add_agent(agent)
        self.time_limit = self.playground.time_limit

        self.game = Engine(self.playground, screen=False)
        self.agent = agent
        assert self.agent in self.game.agents

        # Define action space

        actuators = self.agent.get_all_actuators()
        self.continuous_action_space = continuous_action_space
        self.actions_dict = {}

        if self.continuous_action_space:

            lows = []
            highs = []

            for actuator in actuators:

                if actuator.action_range is ActionSpaces.DISCRETE:
                    lows.append(-1)
                    highs.append(1)

                elif actuator.action_range is ActionSpaces.CONTINUOUS_CENTERED:
                    lows.append(actuator.min)
                    highs.append(actuator.max)

                elif actuator.action_range is ActionSpaces.CONTINUOUS_NOT_CENTERED:
                    lows.append(actuator.min)
                    highs.append(actuator.max)

                else:
                    raise ValueError(f"Action type {actuator.action} unknown")

                # lows.append(actuator.min)
                # highs.append(actuator.max)

            self.action_space = spaces.Box(low=np.array(lows).astype(np.float32),
                                           high=np.array(highs).astype(np.float32))

        else:

            dims = []

            for actuator in actuators:
                if actuator.action_range is ActionSpaces.DISCRETE:
                    dims.append(2)
                elif actuator.action_range is ActionSpaces.CONTINUOUS_NOT_CENTERED:
                    dims.append(2)
                else:
                    raise ValueError(f"Action type {actuator.action} unknown")

            self.action_space = spaces.MultiDiscrete(dims)

        # Define observation space

        # Normalize all sensors to make sure they are in the same range
        height_all_sensors, width_all_sensors, depth_all_sensors = 1, 0, 0
        for sensor in self.agent.sensors:

            if sensor.sensor_modality is SensorTypes.SEMANTIC:
                raise ValueError('Semantic sensors not supported')
            sensor.normalize = True

            if isinstance(sensor.shape, int):
                width_all_sensors = max(width_all_sensors, sensor.shape)
                depth_all_sensors += 1

            elif len(sensor.shape) == 2:
                width_all_sensors = max(width_all_sensors, sensor.shape[0])
                depth_all_sensors += sensor.shape[1]

            else:
                raise NotImplementedError
                # width_all_sensors = max(width_all_sensors, sensor.shape[0])
                # height_all_sensors += sensor.shape[1]

        self.observation_space = spaces.Box(low=0,
                                            high=1,
                                            shape=(1, width_all_sensors,
                                                   depth_all_sensors),
                                            dtype=np.float32)
        self.observations = np.zeros(
            (height_all_sensors, width_all_sensors, depth_all_sensors))

        # Multisteps
        self.multisteps = None
        if multisteps is not None:
            assert isinstance(multisteps, int)
            self.multisteps = multisteps

    @property
    def engine(self):
        return self.game

    def get_current_timestep(self):
        return self.game.elapsed_time

    def step(self, actions):

        # First, send actions to game engint

        actions_to_game_engine = {}
        actions_dict = {}

        # Convert Stable-baselines actions into game engine actions
        for actuator, action in zip(self.agent.get_all_actuators(), actions):
            action_type = actuator.action
            converted_action = action

            # convert discrete action to binry
            if self.continuous_action_space and \
               (action_type is ActionSpaces.DISCRETE):
                converted_action = 0 if converted_action < 0 else 1

            # convert continuous actions in [-1, 1]
            elif (not self.continuous_action_space) and \
                    (action_type is ActionSpaces.CONTINUOUS_CENTERED):
                converted_action = converted_action - 1

            actions_dict[actuator] = converted_action

        actions_to_game_engine[self.agent] = actions_dict

        # Generate actions for other agents
        for agent in self.game.agents:
            if agent is not self.agent:
                actions_to_game_engine[agent] = \
                    agent.controller.generate_actions()

        # Now that we have all ctions, run the engine, and get the observations

        if self.multisteps is None:
            self.game.step(actions_to_game_engine)
        else:
            self.game.multiple_steps(actions_to_game_engine,
                                     n_steps=self.multisteps)

        self.game.update_observations()

        # Concatenate the observations in a format that stable-baselines understands

        sensor_values = []
        for sensor in self.agent.sensors:

            if isinstance(sensor.shape, int):
                sensor_values.append(
                    sensor.sensor_values[np.newaxis, :, np.newaxis])
                # self.observations[0, :sensor.shape,
                #                   current_channel] = sensor.sensor_values[:]
                # current_channel += 1

            elif len(sensor.shape) == 2:
                sensor_values.append(sensor.sensor_values[np.newaxis, :])
                # self.observations[0, :sensor.shape[0],
                #                   current_channel] = sensor.sensor_values[:, :]
                # current_channel += sensor.shape[1]

            else:
                raise NotImplementedError
                # self.observations[:sensor.shape[0], :sensor.
                #                   shape[1], :] = sensor.sensor_values[:, :, :]
                # current_channel += sensor.shape[2]

        self.observations = np.concatenate(sensor_values, axis=2)
        reward = self.agent.reward
        done = self.playground.done or not self.game.game_on

        return (self.observations, reward, done, {})

    def reset(self):

        self.game.reset()
        self.game.elapsed_time = 0
        self.episodes += 1

        return np.zeros(self.observations.shape)

    def render(self, mode='human'):
        if self.video_dir is None:
            return None

        img = self.game.generate_agent_image(self.agent)
        img = (255 * img).astype(np.uint8)

        step_id = self.game.elapsed_time
        video_dir = osp.join(self.video_dir, str(id(self)), str(self.episodes))
        frame_path = osp.join(video_dir, f"f-{step_id:03d}.png")
        if not osp.exists(video_dir):
            os.makedirs(video_dir, exist_ok=True)

        cv2.imwrite(frame_path, img)
        return img

    def close(self):
        self.game.terminate()


def get_sensor_params(sensors_name):
    if sensors_name == 'rgb':
        sensors = [('rgb', {'fov': 180, 'range': 300, 'resolution': 64})]

    elif sensors_name == 'depth':
        sensors = [('depth', {'fov': 180, 'range': 300, 'resolution': 64})]

    elif sensors_name == 'rgb_depth':
        sensors = [('depth', {
            'fov': 180,
            'range': 300,
            'resolution': 64
        }), ('rgb', {
            'fov': 180,
            'range': 300,
            'resolution': 64
        })]

    elif sensors_name == 'rgb_touch':
        sensors = [('rgb', {
            'fov': 180,
            'range': 300,
            'resolution': 64
        }), ('touch', {
            'range': 2,
            'resolution': 64
        })]

    elif sensors_name == 'rgb_depth_touch':
        sensors = [('depth', {
            'fov': 180,
            'range': 300,
            'resolution': 64
        }), ('rgb', {
            'fov': 180,
            'range': 300,
            'resolution': 64
        }), ('touch', {
            'range': 2,
            'resolution': 64
        })]
    elif sensors_name == 'blind':
        sensors = [('blind', {'resolution': 64})]
    else:
        raise ValueError(f"Wrong sensors_name: {sensors_name}")

    return sensors

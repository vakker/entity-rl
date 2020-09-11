import gym
from gym import spaces
from simple_playgrounds.utils import ActionTypes, SensorModality
from simple_playgrounds import Engine
import numpy
from simple_playgrounds.entities.agents import BaseInteractiveAgent, BaseAgent
from simple_playgrounds.entities.agents.agent import Agent
from simple_playgrounds.entities.agents.parts import HolonomicPlatform, Arm, Hand

from simple_playgrounds.controllers import External
from simple_playgrounds.entities.agents.sensors import RgbSensor, DepthSensor, TouchSensor

from stable_baselines.common import set_global_seeds
from environments.rl import *


class Agent_base(BaseInteractiveAgent):

    def __init__(self, sensors):
        super().__init__(controller=External(), allow_overlapping=False)

        for sensor_name, sensor_params in sensors:

            if sensor_name == 'depth':
                self.add_sensor(DepthSensor(anchor=self.base_platform, normalize=True, **sensor_params))

            elif sensor_name == 'rgb':
                self.add_sensor(RgbSensor(anchor=self.base_platform, normalize=True, **sensor_params))

            elif sensor_name == 'touch':
                self.add_sensor(TouchSensor(anchor=self.base_platform, normalize=True, **sensor_params))

import math

class Agent_base_arm(Agent):
    """
        Agent with two Arms.

        Attributes:
            head: Head of the agent
            arm_l: left arm
            arm_l_2: second segment of left arm
            arm_r: right arm
            arm_r_2: second segment of right arm
        """
    def __init__(self, sensors):

        base_agent = HolonomicPlatform(can_eat=False, can_grasp=False, can_activate=False, can_absorb=False, radius=8)

        super().__init__(initial_position=None, base_platform=base_agent, controller = External(), allow_overlapping = False)

        self.arm_r = Arm(base_agent, [8, 0], angle_offset=-math.pi / 2, rotation_range=math.pi, width_length=[3, 10])
        self.add_body_part(self.arm_r)

        self.arm_r_2 = Arm(self.arm_r, self.arm_r.extremity_anchor_point, angle_offset=math.pi / 2, rotation_range=math.pi, width_length=[3, 10])
        self.add_body_part(self.arm_r_2)

        self.hand_r = Hand(self.arm_r_2, self.arm_r_2.extremity_anchor_point, radius=5, rotation_range=math.pi, can_grasp = True)
        self.add_body_part(self.hand_r)

        for sensor_name, sensor_params in sensors:

            if sensor_name == 'depth':
                self.add_sensor(DepthSensor(anchor=self.base_platform, normalize=True, **sensor_params))

            elif sensor_name == 'rgb':
                self.add_sensor(RgbSensor(anchor=self.base_platform, normalize=True, **sensor_params))

            elif sensor_name == 'touch':
                self.add_sensor(TouchSensor(anchor=self.hand_r, normalize=True, **sensor_params))


class PlaygroundEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, playground, agent, continuous_action_space = True, multisteps = None):
        """
        Args:
            game_engine: Engine to run the

        """

        super().__init__()

        self.time_limit = playground.time_limit

        self.game = Engine(playground, replay=False, screen=False)
        self.agent = agent
        assert self.agent in self.game.agents

        # Define action space

        actions = self.agent.get_all_available_actions()
        self.continuous_action_space = continuous_action_space
        self.actions_dict = {}

        if self.continuous_action_space:

            lows = []
            highs = []

            for action in actions:

                if action.action_type is ActionTypes.DISCRETE:
                    lows.append(-1)
                    highs.append(1)

                elif action.action_type is ActionTypes.CONTINUOUS_CENTERED:
                    lows.append(action.min)
                    highs.append(action.max)

                elif action.action_type is ActionTypes.CONTINUOUS_NOT_CENTERED:
                    lows.append(action.min)
                    highs.append(action.max)

                # lows.append(action.min)
                # highs.append(action.max)

                if action.body_part not in self.actions_dict:
                    self.actions_dict[action.body_part] = {}

                if action.action not in self.actions_dict[action.body_part]:
                    self.actions_dict[action.body_part][action.action] = 0

            self.action_space = spaces.Box(low=numpy.array(lows), high=numpy.array(highs))

        else:

            dims = []

            for action in actions:
                if action.action_type is ActionTypes.DISCRETE:
                    dims.append(2)
                elif action.action_type is ActionTypes.CONTINUOUS_CENTERED:
                    dims.append(3)
                elif action.action_type is ActionTypes.CONTINUOUS_NOT_CENTERED:
                    dims.append(2)

            self.action_space = spaces.MultiDiscrete(dims)

        # Define observation space

        # Normalize all sensors to make sure they are in the same range
        width_all_sensors, height_all_sensors = 0, 0
        for sensor in self.agent.sensors:

            if sensor.sensor_modality is SensorModality.SEMANTIC:
                raise ValueError( 'Semantic sensors not supported')
            sensor.normalize = True

            if isinstance(sensor.shape, int):
                width_all_sensors = max(width_all_sensors, sensor.shape)
                height_all_sensors += 1

            elif len(sensor.shape) == 2:
                width_all_sensors = max(width_all_sensors, sensor.shape[0])
                height_all_sensors += 1

            else:
                width_all_sensors = max(width_all_sensors, sensor.shape[0])
                height_all_sensors += sensor.shape[1]

        self.observation_space = spaces.Box(low=0, high=1, shape=(height_all_sensors, width_all_sensors, 3), dtype=numpy.float32)
        self.observations = numpy.zeros((height_all_sensors, width_all_sensors, 3))

        # Multisteps
        self.multisteps = None
        if multisteps is not None:
            assert isinstance(multisteps, int)
            self.multisteps = multisteps

    def get_current_timestep(self):
        return self.game.elapsed_time


    def step(self, action):

        # First, send actions to game engint

        actions_to_game_engine = {}

        # Convert Stable-baselines actions into game engine actions
        for index_action, available_action in enumerate(self.agent.get_all_available_actions()):

            body_part = available_action.body_part
            action_ = available_action.action
            action_type = available_action.action_type

            converted_action = action[index_action]

            # convert discrete action to binry
            if self.continuous_action_space and action_type is ActionTypes.DISCRETE:
                converted_action = 0 if converted_action < 0 else 1

            # convert continuous actions in [-1, 1]
            elif not self.continuous_action_space and action_type is ActionTypes.CONTINUOUS_CENTERED:
                converted_action = converted_action - 1

            self.actions_dict[body_part][action_] = converted_action

        actions_to_game_engine[self.agent.name] = self.actions_dict

        # Generate actions for other agents
        for agent in self.game.agents:
            if agent is not self.agent:
                actions_to_game_engine[agent.name] = agent.controller.generate_actions()

        # Now that we have all ctions, run the engine, and get the observations

        if self.multisteps is None:
            reset, terminate = self.game.step(actions_to_game_engine)
        else:
            reset, terminate = self.game.multiple_steps(actions_to_game_engine, n_steps=self.multisteps)

        self.game.update_observations()



        # Concatenate the observations in a format that stable-baselines understands

        current_height = 0
        for sensor in self.agent.sensors:

            if isinstance(sensor.shape, int):
                self.observations[current_height, :sensor.shape, 0] = sensor.sensor_value[:]
                current_height += 1

            elif len(sensor.shape) == 2:
                self.observations[current_height, :sensor.shape[0], :] = sensor.sensor_value[:, :]
                current_height += 1

            else:
                self.observations[:sensor.shape[0], :sensor.shape[1], :] = sensor.sensor_value[:, :, :]
                current_height += sensor.shape[0]


        reward = self.agent.reward

        if reset or terminate:
            done = True

        else:
            done = False

        return (self.observations, reward, done, {})


    def reset(self):

        self.game.reset()
        self.game.elapsed_time = 0

        return numpy.zeros(self.observations.shape)

    def render(self, mode='human'):
        img = self.game.generate_topdown_image()
        return img

    def close(self):
        self.game.terminate()


import random

def make_vector_env(playground_name, agent_type, sensors, rank, multisteps = None, seed=0):
    """
    Utility function for multiprocessed env.

    Args:
        pg: Instance of a Playground
        Ag: Agent
    """

    def _init():

        random.seed(seed)
        numpy.random.seed(seed)

        playground = PlaygroundRegister.playgrounds[playground_name]()

        random.seed(seed+rank)
        numpy.random.seed(seed+rank)

        if agent_type == 'base':
            agent = Agent_base(sensors)

        elif agent_type == 'arm':
            agent = Agent_base_arm(sensors)

        playground.add_agent(agent)

        custom_env = PlaygroundEnv(playground, agent, multisteps=multisteps, continuous_action_space=True)

        return custom_env


    return _init


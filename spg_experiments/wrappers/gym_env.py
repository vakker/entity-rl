import random

import gym
import numpy
from gym import spaces
from simple_playgrounds import Engine
from simple_playgrounds.agents import agents, controllers, sensors
from simple_playgrounds.agents.parts.platform import ForwardBackwardPlatform
from simple_playgrounds.playgrounds import PlaygroundRegister
from simple_playgrounds.utils import ActionTypes, SensorModality


class PlaygroundEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, config):
        super().__init__()

        playground_name = config['playground_name']
        agent_type = config['agent_type']
        sensors_name = config['sensors_name']
        seed = config.get('seed', 0)
        continuous_action_space = config.get('continuous_action_space', True)
        multisteps = config.get('multisteps')
        controller = config.get('controller', controllers.External())

        self.playground = PlaygroundRegister.playgrounds[playground_name]()

        seed = (seed + id(self)) % (2**32)
        random.seed(seed)
        numpy.random.seed(seed)

        if agent_type == 'base':
            agent_cls = agents.BaseAgent
        elif agent_type == 'arm':
            agent_cls = agents.FullAgent
        else:
            raise ValueError(f"Wrong agent_type: {agent_type}")

        agent = agent_cls(platform=ForwardBackwardPlatform,
                          controller=controllers.External())

        for sensor_name, sensor_params in get_sensor_params(sensors_name):
            if sensor_name == 'depth':
                agent.add_sensor(
                    sensors.Lidar(anchor=agent.base_platform,
                                  normalize=True,
                                  **sensor_params))

            elif sensor_name == 'rgb':
                agent.add_sensor(
                    sensors.RgbCamera(anchor=agent.base_platform,
                                      normalize=True,
                                      **sensor_params))

            elif sensor_name == 'touch':
                agent.add_sensor(
                    sensors.Touch(anchor=agent.base_platform,
                                  normalize=True,
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

                if actuator.action_type is ActionTypes.DISCRETE:
                    lows.append(-1)
                    highs.append(1)

                elif actuator.action_type is ActionTypes.CONTINUOUS_CENTERED:
                    lows.append(actuator.min)
                    highs.append(actuator.max)

                elif actuator.action_type is ActionTypes.CONTINUOUS_NOT_CENTERED:
                    lows.append(actuator.min)
                    highs.append(actuator.max)

                lows.append(actuator.min)
                highs.append(actuator.max)

            self.action_space = spaces.Box(low=numpy.array(lows), high=numpy.array(highs))

        else:

            dims = []

            for actuator in actuators:
                if actuator.action_type is ActionTypes.DISCRETE:
                    dims.append(2)
                elif actuator.action_type is ActionTypes.CONTINUOUS_NOT_CENTERED:
                    dims.append(2)

            self.action_space = spaces.MultiDiscrete(dims)

        # Define observation space

        # Normalize all sensors to make sure they are in the same range
        width_all_sensors, height_all_sensors = 0, 0
        for sensor in self.agent.sensors:

            if sensor.sensor_modality is SensorModality.SEMANTIC:
                raise ValueError('Semantic sensors not supported')
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

        self.observation_space = spaces.Box(low=0,
                                            high=1,
                                            shape=(height_all_sensors, width_all_sensors,
                                                   3),
                                            dtype=numpy.float32)
        self.observations = numpy.zeros((height_all_sensors, width_all_sensors, 3))

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
            action_type = actuator.action_type
            converted_action = action

            # convert discrete action to binry
            if self.continuous_action_space and action_type is ActionTypes.DISCRETE:
                converted_action = 0 if converted_action < 0 else 1

            # convert continuous actions in [-1, 1]
            elif (not self.continuous_action_space) and (action_type is
                                                         ActionTypes.CONTINUOUS_CENTERED):
                converted_action = converted_action - 1

            actions_dict[actuator] = converted_action

        actions_to_game_engine[self.agent] = actions_dict

        # Generate actions for other agents
        for agent in self.game.agents:
            if agent is not self.agent:
                actions_to_game_engine[agent] = agent.controller.generate_actions()

        # Now that we have all ctions, run the engine, and get the observations

        if self.multisteps is None:
            terminate = self.game.step(actions_to_game_engine)
        else:
            terminate = self.game.multiple_steps(actions_to_game_engine,
                                                 n_steps=self.multisteps)

        self.game.update_observations()

        # Concatenate the observations in a format that stable-baselines understands

        current_height = 0
        for sensor in self.agent.sensors:

            if isinstance(sensor.shape, int):
                self.observations[current_height, :sensor.shape,
                                  0] = sensor.sensor_value[:]
                current_height += 1

            elif len(sensor.shape) == 2:
                self.observations[
                    current_height, :sensor.shape[0], :] = sensor.sensor_values[:, :]
                current_height += 1

            else:
                self.observations[:sensor.shape[0], :sensor.
                                  shape[1], :] = sensor.sensor_values[:, :, :]
                current_height += sensor.shape[0]

        reward = self.agent.reward
        done = self.playground.done or terminate

        return (self.observations, reward, done, {})

    def reset(self):

        self.game.reset()
        self.game.elapsed_time = 0

        return numpy.zeros(self.observations.shape)

    def render(self, mode='human'):
        img = self.game.generate_playground_image()
        return img

    def close(self):
        self.game.terminate()


def get_sensor_params(sensors_name):
    if sensors_name == 'rgb':
        sensors = [('rgb', {'fov': 180, 'range': 300, 'resolution': 64})]

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
    else:
        raise ValueError(f"Wrong sensors_name: {sensors_name}")

    return sensors

from simple_playgrounds.controllers import External
from simple_playgrounds.entities.agents import BaseInteractiveAgent
from simple_playgrounds.entities.agents.sensors import RgbSensor, DepthSensor, TouchSensor
from simple_playgrounds.playgrounds.collection import *
from stable_baselines.common.vec_env import SubprocVecEnv
from wrappers.gym_env import make_vector_env
from wrappers.stable_wrappers import CustomPolicy
from stable_baselines import PPO2

class MyAgent(BaseInteractiveAgent):

    def __init__(self, sensors):
        super().__init__(controller=External(), allow_overlapping=False)

        for sensor_name, sensor_params in sensors:

            if sensor_name == 'depth':
                self.add_sensor(DepthSensor(anchor=self.base_platform, normalize=True, **sensor_params))

            elif sensor_name == 'rgb':
                self.add_sensor(RgbSensor(anchor=self.base_platform, normalize=True, **sensor_params))

            elif sensor_name == 'touch':
                self.add_sensor(TouchSensor(anchor=self.base_platform, normalize=True, **sensor_params))


def eval(env, model):

    # model.learn(total_timesteps=int(1e5))
    rew = 0
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        rew += sum(rewards) / 4.0
        done = any(dones)

    return rew

def train_eval( sensors,
                total_timesteps_training,
                n_multisteps,
                playground_name,
                timesteps_eval,
                exp_name,
                ):



    pg = PlaygroundRegister.playgrounds[playground_name]()
    agent = MyAgent(sensors)

    train_envs = SubprocVecEnv([make_vector_env(pg, agent, multisteps=n_multisteps) for i in range(4)], start_method='fork')
    test_env = SubprocVecEnv([make_vector_env(pg, agent, multisteps=n_multisteps) for i in range(4)], start_method='fork')

    model = PPO2(CustomPolicy, train_envs, policy_kwargs={'observation_shape': agent.get_visual_sensor_shapes()}, verbose=0)

    assert pg.time_limit is not None

    n_training_steps = int(total_timesteps_training / timesteps_eval)

    fname = 'logs/' + exp_name + '.dat'

    # Eval untrained
    rew = eval(test_env, model)

    with open(fname, 'a') as f:
        f.write('0;'+str(rew)+'\n')

    for i in range(1, n_training_steps+1):

        model.learn(timesteps_eval)

        rew = eval(test_env, model)

        with open(fname, 'a') as f:
            f.write(str(i*timesteps_eval)+'_' + str(rew) + '\n')

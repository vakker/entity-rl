from simple_playgrounds.controllers import External
from simple_playgrounds.entities.agents import BaseInteractiveAgent
from simple_playgrounds.entities.agents.sensors import RgbSensor, DepthSensor, TouchSensor
from environments.rl import *

from stable_baselines.common.vec_env import SubprocVecEnv
from wrappers.gym_env import make_vector_env
from wrappers.stable_wrappers import CustomPolicy
from stable_baselines import PPO2
import yaml

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


def eval(env, model, episodes_eval):

    # model.learn(total_timesteps=int(1e5))
    all_rewards = []
    all_times = []

    for i in range(episodes_eval):
        obs = env.reset()

        # Only keep one episode, don't take into account restart.
        done = [False, False, False, False]
        end_time = [0, 0, 0, 0]
        rew = [0, 0, 0, 0]

        while not all(done):

            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)

            for env_index in range(4):

                if not done[env_index]:
                    rew[env_index] += rewards[env_index]

                if dones[env_index] == True :
                    done[env_index] = True

                if not done[env_index]:
                    end_time[env_index] = env.env_method('get_current_timestep')[env_index]



        all_rewards += rew
        all_times += end_time

    result = {}
    for i in range(len(all_rewards)):
        result[i] = { 'duration': all_times[i] , 'cumulated_reward': all_rewards[i]}

    return result

def train_and_eval( sensors,
                total_timesteps_training,
                n_multisteps,
                playground_name,
                freq_eval,
                episodes_eval,
                exp_name,
                ):


    results = {}

    pg = PlaygroundRegister.playgrounds[playground_name]()
    agent = MyAgent(sensors)

    train_envs = SubprocVecEnv([make_vector_env(pg, agent, multisteps=n_multisteps) for i in range(4)], start_method='fork')
    test_env = SubprocVecEnv([make_vector_env(pg, agent, multisteps=n_multisteps) for i in range(4)], start_method='fork')

    model = PPO2(CustomPolicy, train_envs, policy_kwargs={'observation_shape': agent.get_visual_sensor_shapes()}, verbose=0)

    assert pg.time_limit is not None

    n_training_steps = int(total_timesteps_training / freq_eval)


    # Eval untrained
    res = eval(test_env, model, episodes_eval)
    results[0] = res
    print(res)

    for i in range(1, n_training_steps+1):

        model.learn(freq_eval)

        res = eval(test_env, model, episodes_eval)
        results[ i * freq_eval] = res
        print(res)

    test_env.close()
    train_envs.close()

    for time, res in results.items():
        print(time, res)

    fname = 'logs/' + exp_name + '.dat'

    with open(fname, 'w') as f:
        yaml.dump(results, f)

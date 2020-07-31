
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from wrappers.gym_env import make_vector_env
from wrappers.stable_wrappers import CustomPolicy
from stable_baselines import PPO2
import yaml



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


from .gym_env import MyAgent

import random


def train_and_eval( agent_type, sensors,
                total_timesteps_training,
                n_multisteps,
                playground_name,
                freq_eval,
                episodes_eval,
                exp_name,
                ):


    results = {}

    agent = MyAgent( sensors)

    seed = random.randint(0,1000)

    train_envs = SubprocVecEnv([make_vector_env(playground_name, sensors, multisteps=n_multisteps, seed=seed) for i in range(4)], start_method='spawn')
    test_env = SubprocVecEnv([make_vector_env(playground_name, sensors, multisteps=n_multisteps, seed=seed) for i in range(4)], start_method='spawn')

    # train_envs = DummyVecEnv([make_vector_env(playground_name, sensors, multisteps=n_multisteps) for i in range(4)])
    # test_env = DummyVecEnv([make_vector_env(playground_name, sensors, multisteps=n_multisteps) for i in range(4)])

    model = PPO2(CustomPolicy, train_envs, policy_kwargs={'observation_shape': agent.get_visual_sensor_shapes()}, verbose=0)

    time_limit = train_envs.get_attr('time_limit', indices=0)
    assert time_limit is not None

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

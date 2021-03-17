import argparse
from datetime import datetime

import ray
from ray import tune
from ray.rllib.agents import ppo
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune import CLIReporter
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray.tune.suggest.variant_generator import grid_search

from spg_experiments.wrappers import models
from spg_experiments.wrappers.gym_env import PlaygroundEnv

parser = argparse.ArgumentParser()
parser.add_argument("--logdir", type=str, default="logs")
parser.add_argument("--run", type=str, default="PPO")
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--local", action="store_true")
parser.add_argument("--gpu", action="store_true")
parser.add_argument("--monitor", action="store_true")
parser.add_argument("--stop-iters", type=int, default=50)
parser.add_argument("--num-workers", type=int, default=5)
parser.add_argument("--stop-timesteps", type=int, default=1000000)
parser.add_argument("--stop-reward", type=float, default=75)


def exp_name(prefix):
    return prefix + '.' + datetime.now().strftime("%Y-%m-%d.%H:%M:%S")


class E(dict):
    def keys(self):
        return []


def trial_str_creator(trial):
    return f'trial-{trial.trial_id}'


def main(args):
    ray.init(local_mode=args.local)

    config = {
        "num_workers": args.num_workers,  # parallelism
        # "evaluation_config": {
        #     "env_config": {"save_location": "scenes"},
        #     # "explore": False,
        # },
        "evaluation_interval": int(1e5),
        "env": PlaygroundEnv,
        # "monitor": args.monitor,
        "output": "logdir",
        # "render_env": args.monitor,
        # "record_env": args.monitor,
        "env_config": {
            "agent_type": "base",
            # "index_exp": grid_search([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
            "playground_name": grid_search([
                "spg_endgoal_cue",
                "spg_endgoal_9rooms",
                "spg_dispenser_6rooms",
                "spg_coinmaster_singleroom",
            ]),
            "sensors_name": grid_search(["rgb", "rgb_depth rgb_touch",
                                         "rgb_depth_touch"]),
            "multistep": ([0, 2, 3, 4])
        },
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": 0.5 if args.gpu else 0,
        # "model": {
        #     "custom_model": "my_model",
        #     "vf_share_layers": True,
        #     "fcnet_hiddens": [1],
        # },
        # # "lr": grid_search([1e-2, 1e-4, 1e-6]),  # try different lrs
        "framework": "torch",
        "gamma": 0.99,  # checked
        "lr": 0.00025,  # checked
        "lambda": 0.95,  # checked
        "kl_coeff": 0.5,  # ?
        "clip_rewards": False,  # checked?
        "clip_param": 0.2,  # checked?
        "grad_clip": 0.5,  # checked
        "vf_clip_param": 10000,  # checked, it's None in SB
        "vf_loss_coeff": 0.5,  # checked
        "entropy_coeff": grid_search([0.05, 0.01, 0.005, 0.001]),  # checked
        # "timesteps_per_iteration": 500,
        # "train_batch_size": 500,
        # "sgd_minibatch_size": 100,
        "num_sgd_iter": 4,  # checked
        "num_envs_per_worker": 1,
        # "timesteps_per_iteration": 128 * 4,  # this seems redundant
        "train_batch_size": 128 * 4,  # checked, but check the *4
        "sgd_minibatch_size": 128,  # train_batch_size/4
        "num_sgd_iter": 4,  # checked, not sure
        # "num_envs_per_worker": 5,
        "batch_mode": "truncate_episodes",
        "observation_filter": "NoFilter",
        "model": {
            "custom_model": "custom-fc",
            # "vf_share_layers": True,
            # "custom_model": TorchCustomModel,
            # "dim": 300,
            # "conv_filters": [
            #     [16, [4, 4], 4],
            #     [32, [4, 4], 2],
            #     [64, [4, 4], 2],
            #     [512, [11, 11], 2],
            #     [512, [10, 10], 1],
            # ],
        },
    }

    stop = {
        # "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        # "episode_reward_mean": args.stop_reward,
    }

    name = exp_name('PPO')
    reporter = CLIReporter(parameter_columns=E({"_": "_"}))
    results = tune.run(
        args.run,
        config=config,
        stop=stop,
        local_dir=args.logdir,
        checkpoint_at_end=True,
        checkpoint_freq=1,
        trial_name_creator=trial_str_creator,
        trial_dirname_creator=trial_str_creator,
        progress_reporter=reporter,
        name=name,
        # max_failures=1,
        fail_fast=True)

    # if args.as_test:
    #     check_learning_achieved(results, args.stop_reward)
    # ray.shutdown()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

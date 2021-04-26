import argparse
from datetime import datetime

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.suggest.variant_generator import grid_search

from spg_experiments import models
from spg_experiments.gym_env import PlaygroundEnv


def exp_name(prefix):
    return prefix + '.' + datetime.now().strftime("%Y-%m-%d.%H:%M:%S")


class E(dict):
    def keys(self):
        return []


def trial_str_creator(trial):
    params = {
        k.split('/')[-1]: p[-1] if isinstance(p, list) else str(p)
        for k, p in trial.evaluated_params.items()
    }
    name = '-'.join([f'{k}:{p}' for k, p in params.items()])
    return f'trial-{name}'


def main(args):
    ray.init(local_mode=args.local)

    config = {
        "num_workers": args.num_workers,  # parallelism
        "num_envs_per_worker": 2,
        "num_cpus_per_worker": 0.5,
        "evaluation_num_workers": args.num_workers,
        # "evaluation_config": {
        # },
        "evaluation_interval": 10,
        "env": PlaygroundEnv,
        "output": "logdir",
        "env_config": {
            "agent_type": "base",
            # "index_exp": grid_search([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
            "playground_name": grid_search([
                ["foraging", "candy_collect"],
                ["foraging", "candy_fireballs"],
                ['navigation', 'endgoal_cue'],
                ['sequential', 'door_dispenser_coin'],
            ]),
            "sensors_name": grid_search([
                "blind",
                "rgb",
                "depth",
                "rgb_depth",
                "rgb_touch",
                "rgb_depth_touch",
            ]),
            # "multisteps": grid_search([0, 2, 3, 4])
            # "multisteps": 0
        },
        "num_gpus": 0.5 if args.gpu else 0,
        "framework": "torch",
        "gamma": grid_search([0.1, 0.2, 0.5, 0.8, 0.99]),  # checked
        "lr": grid_search([0.001, 0.0001, 0.00001]),
        "lambda": 0.95,  # checked
        # "kl_coeff": 0.5,  # ?
        "clip_rewards": False,
        "clip_param": 0.2,  # checked?
        "grad_clip": 0.5,  # checked
        # "vf_clip_param": 10,  # checked, it's None in SB, 10 in RLlib
        "vf_loss_coeff": 0.0001,  # checked
        "entropy_coeff": grid_search([0.05, 0.01, 0.005, 0.001]),  # checked
        "train_batch_size": 128 * 10 * 8,  # checked, but check the *4*2
        "sgd_minibatch_size": 128,  # could be larger
        "num_sgd_iter": 4,  # checked?
        "batch_mode": "truncate_episodes",
        "observation_filter": "NoFilter",
        "model": {
            "custom_model": "vision-1d",
            "conv_filters": [
                [64, 5, 3],
                [64, 3, 2],
                [64, 3, 2],
                [128, 3, 2],
                [128, 3, 2],
                # [128, 3, 2],
            ],
            "use_lstm": grid_search([True, False]),
        },
    }

    stop = {"timesteps_total": args.stop_timesteps}
    if args.stop_iters:
        stop.update({"training_iteration": args.stop_iters})
    if args.stop_reward:
        stop.update({"episode_reward_mean": args.stop_reward})

    name = exp_name('PPO')
    reporter = CLIReporter(parameter_columns=E({"_": "_"}))
    results = tune.run(
        args.run,
        config=config,
        stop=stop,
        local_dir=args.logdir,
        checkpoint_at_end=True,
        checkpoint_freq=1,
        keep_checkpoints_num=2,
        trial_name_creator=trial_str_creator,
        trial_dirname_creator=trial_str_creator,
        progress_reporter=reporter,
        name=name,
        # max_failures=5,
        fail_fast=True,
        verbose=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--run", type=str, default="PPO")
    parser.add_argument("--as-test", action="store_true")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--monitor", action="store_true")
    parser.add_argument("--num-workers", type=int, default=5)
    parser.add_argument("--stop-timesteps", type=int, default=1000000)
    parser.add_argument("--stop-iters", type=int)
    parser.add_argument("--stop-reward", type=float)

    args = parser.parse_args()
    main(args)

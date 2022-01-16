import json
import os
from collections import defaultdict
from os import path as osp

import numpy as np
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.tune.logger import LoggerCallback
from ray.tune.result import TIMESTEPS_TOTAL, TRAINING_ITERATION


class CustomCallbacks(DefaultCallbacks):
    # pylint: disable=arguments-differ,unused-argument

    def on_episode_start(
        self, *, worker, base_env, policies, episode, env_index, **kwargs
    ):
        episode.media["episode_data"] = defaultdict(list)
        episode.user_data = {"final": {}, "running": defaultdict(list)}

    def on_episode_step(self, *, worker, base_env, episode, env_index, **kwargs):
        # Running metrics -> keep all values
        # Final metrics -> only keep the current value
        for data_type, data_subset in episode.user_data.items():
            data = episode.last_info_for().get("data", {}).get(data_type, {})
            for name, value in data.items():
                if data_type == "running":
                    data_subset[name].append(value)
                else:
                    data_subset[name] = value

        # Arbitrary episode media
        media = episode.last_info_for().get("media", {})
        for name, value in media.items():
            episode.media["episode_data"][name].append(value)

    def on_episode_end(
        self, *, worker, base_env, policies, episode, env_index, **kwargs
    ):
        for name, value in episode.media["episode_data"].items():
            episode.media["episode_data"][name] = np.array(value).tolist()

        for data_type, data_subset in episode.user_data.items():
            for name, value in data_subset.items():
                if data_type == "running":
                    episode.custom_metrics[name + "_avg"] = np.mean(value)
                    episode.custom_metrics[name + "_sum"] = np.sum(value)
                    episode.hist_data[name] = value
                else:
                    episode.custom_metrics[name] = value
                    episode.hist_data[name] = [value]


class DataLoggerCallback(LoggerCallback):
    def __init__(self):
        self._trial_continue = {}
        self._trial_local_dir = {}

    def log_trial_start(self, trial):
        trial.init_logdir()
        self._trial_local_dir[trial] = osp.join(trial.logdir, "episode_data")
        os.makedirs(self._trial_local_dir[trial], exist_ok=True)

    def log_trial_result(self, iteration, trial, result):
        if "episode_data" not in result["episode_media"]:
            return

        step = result.get(TIMESTEPS_TOTAL) or result[TRAINING_ITERATION]

        num_episodes = result["episodes_this_iter"]
        data = result["episode_media"]["episode_data"]
        episode_data = data[-num_episodes:]

        if "evaluation" in result:
            data = result["evaluation"]["episode_media"]["episode_data"]
            episode_data += data[-num_episodes:]

        data_file_name = osp.join(self._trial_local_dir[trial], f"data-{step:08d}.json")
        with open(data_file_name, "w") as data_file:
            json.dump(episode_data, data_file)

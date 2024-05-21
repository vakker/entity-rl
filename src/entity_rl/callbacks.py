import json
import logging
import os
import traceback
from collections import defaultdict
from os import path as osp
from typing import Dict, Optional

import aim
import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune.experiment import Trial
from ray.tune.logger import LoggerCallback
from ray.tune.result import TIME_TOTAL_S, TIMESTEPS_TOTAL, TRAINING_ITERATION
from ray.tune.utils import flatten_dict
from ray.util.debug import log_once

logger = logging.getLogger(__name__)


class CustomCallbacks(DefaultCallbacks):
    # pylint: disable=arguments-differ,unused-argument

    def on_episode_start(
        self, *, worker, base_env, policies, episode, env_index, **kwargs
    ):
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
        if media:
            if "episode_data" not in episode.media:
                episode.media["episode_data"] = {}

        for name, value in media.items():
            episode.media["episode_data"][name].append(value)

    def on_episode_end(
        self, *, worker, base_env, policies, episode, env_index, **kwargs
    ):
        for name, value in episode.media.get("episode_data", {}).items():
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


class AimLoggerCallback(LoggerCallback):
    VALID_HPARAMS = (str, bool, int, float, list, type(None))
    VALID_NP_HPARAMS = (np.bool8, np.float32, np.float64, np.int32, np.int64)
    VALID_SUMMARY_TYPES = [int, float, np.float32, np.float64, np.int32, np.int64]

    def __init__(
        self,
        experiment_name: Optional[str] = "ray-tune",
        repo: Optional[str] = None,
    ):
        self.experiment_name = experiment_name
        self.repo = repo
        self._trial_runs = {}

    def log_trial_start(self, trial: Trial):
        # TODO: implement restart logic

        # Create run if not already exists.
        if trial not in self._trial_runs:
            run = aim.Run(
                experiment=self.experiment_name,
                repo=self.repo,
                system_tracking_interval=None,
                capture_terminal_logs=False,
            )
            # For backward compatibility:
            run.name = "-".join(str(trial).split("-")[:2])
            # run.name = str(trial)
            self._trial_runs[trial] = run

        else:
            run = self._trial_runs[trial]

        # Log the config parameters.
        config = trial.config.copy()
        run["hparams"] = self._clear_hparams(config)

    def _clear_hparams(self, hparams: Dict):
        flat_params = flatten_dict(hparams, delimiter="_")
        scrubbed_params = {
            k: v for k, v in flat_params.items() if isinstance(v, self.VALID_HPARAMS)
        }

        np_params = {
            k: v.tolist()
            for k, v in flat_params.items()
            if isinstance(v, self.VALID_NP_HPARAMS)
        }

        scrubbed_params.update(np_params)

        removed = {
            k: v
            for k, v in flat_params.items()
            if not isinstance(v, self.VALID_HPARAMS + self.VALID_NP_HPARAMS)
        }
        if removed:
            logger.info(
                "Removed the following hyperparameter values when "
                "logging to tensorboard: %s",
                str(removed),
            )

        return scrubbed_params

    def log_trial_result(self, iteration: int, trial: Trial, result: Dict):
        if trial not in self._trial_runs:
            self.log_trial_start(trial)

        step = result.get(TIMESTEPS_TOTAL) or result[TRAINING_ITERATION]

        flat_result = self._clear_result(result)
        path = ["ray", "tune"]

        for attr, value in flat_result.items():
            full_attr = "/".join(path + [attr])
            full_attr = full_attr.replace("/", "_")
            full_attr = full_attr.replace(".", "_")

            if isinstance(value, tuple(self.VALID_SUMMARY_TYPES)) and not np.isnan(
                value
            ):
                self._trial_runs[trial].track(
                    value=value,
                    name=full_attr,
                    step=step,
                )

            elif (isinstance(value, list) and len(value) > 0) or (
                isinstance(value, np.ndarray) and value.size > 0
            ):
                try:
                    if isinstance(value, np.ndarray) and value.ndim > 1:
                        # Must be something weird that's not supported
                        raise ValueError()

                    # Otherwise it's some distribution
                    d = aim.Distribution(
                        samples=value,
                        bin_count=50,
                    )
                    self._trial_runs[trial].track(
                        value=d,
                        name=full_attr,
                        step=step,
                    )

                except (ValueError, TypeError):
                    if log_once("invalid_aim_value"):
                        print(traceback.format_exc())
                        logger.warning(
                            "You are trying to log an invalid value (%s=%s) via %s!",
                            full_attr,
                            value,
                            type(self).__name__,
                        )

    def _clear_result(self, result: Dict):
        tmp = result.copy()
        for k in ["config", "pid", "timestamp", TIME_TOTAL_S, TRAINING_ITERATION]:
            if k in tmp:
                del tmp[k]  # not useful to log these

        flat_result = flatten_dict(tmp, delimiter=".")

        return flat_result

    def log_trial_end(self, trial: Trial, failed: bool = False):
        # if not failed:
        #     self._trial_runs[trial].report_successful_finish()

        if trial in self._trial_runs:
            self._trial_runs[trial].close()
            del self._trial_runs[trial]

import json
from datetime import datetime
from os import path as osp

import yaml
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.ax import AxSearch
from ray.tune.suggest.hyperopt import HyperOptSearch

from . import gym, models
from .callbacks import CustomCallbacks


class E(dict):
    def keys(self):
        return []


def register():
    register_env("spg-v0", gym.PlaygroundEnv)
    # register_env("spg-graph", gym.PlaygroundEnv)
    # register_env("spg-set", gym.PlaygroundEnv)

    ModelCatalog.register_custom_model("fc-net", models.FcNetwork)
    ModelCatalog.register_custom_model("cnn-net", models.CnnNetwork)
    ModelCatalog.register_custom_model("attn-net", models.AttnNetwork)
    ModelCatalog.register_custom_model("gnn-net", models.GnnNetwork)


def get_env_creator(env_name):
    if env_name == "spg-v0":
        return gym.PlaygroundEnv

    raise NotImplementedError()


def exp_name(prefix):
    return prefix + "." + datetime.now().strftime("%Y-%m-%d.%H:%M:%S")


def trial_str_creator(trial):
    return f"trial-{trial.trial_id}"


def load_dict(dict_path):
    _, ext = osp.splitext(dict_path)
    with open(dict_path, "r") as stream:
        dict_str = stream.read()

        if ext in [".json"]:
            yaml_dict = json.loads(dict_str)
        elif ext in [".yml", ".yaml"]:
            yaml_dict = yaml.safe_load(dict_str)
        else:
            raise RuntimeError("No configs found")
    return yaml_dict


def get_configs(log_dir):
    assert osp.isdir(log_dir), "Log dir does not exists."
    # project_dir = get_project_dir()

    if osp.exists(osp.join(log_dir, "conf.yaml")):
        exp_configs = load_dict(osp.join(log_dir, "conf.yaml"))
        # symbols = {'p': project_dir,
        #            'l': log_dir,
        #            'e': log_dir.split('/')[-1]}
        # exp_configs = resolve_symbols(exp_configs, symbols)
        return exp_configs

    if osp.exists(osp.join(log_dir, "params.json")):
        exp_configs = load_dict(osp.join(log_dir, "params.json"))
        return exp_configs

    raise RuntimeError("No configs found")


def update_recursive(d, target_key, value):
    if isinstance(target_key, list):
        pass
    elif isinstance(target_key, str):
        target_key = target_key.split("/")
    else:
        raise ValueError(f"Wrong target key: {target_key}")

    if len(target_key) == 1:
        d[target_key[0]] = value
    else:
        update_recursive(d[target_key[0]], target_key[1:], value)


def parse_tune_configs(configs):
    exp = configs["base"]
    for k, v in configs.get("tune", {}).items():
        value = getattr(tune, v["type"])(*v["args"])
        update_recursive(exp, k, value)

    return exp


def get_tune_params(args):
    conf_yaml = get_configs(args["logdir"])
    if args["tune"]:
        configs = parse_tune_configs(conf_yaml)
    else:
        configs = conf_yaml["base"]

    # Hack to make serving easier
    configs["env_config"]["__run"] = conf_yaml["run"]
    configs["env_config"]["logdir"] = osp.realpath(args["logdir"])

    extra = {
        "num_workers": args["num_workers"],
        "evaluation_config": {
            "env_config": {"is_eval": True},
        },
        "evaluation_interval": args["eval_int"],
        "evaluation_num_episodes": 10,
        "num_cpus_per_worker": 0.5 if args["eval_int"] else 1,
        "evaluation_num_workers": args["num_workers"] if args["eval_int"] else 0,
        "num_gpus": 0 if args["no_gpu"] else conf_yaml["gpu_per_worker"],
        "framework": "torch",
        "num_envs_per_worker": 1,
        "batch_mode": "truncate_episodes",
        "observation_filter": "NoFilter",
    }

    configs.update(extra)
    configs["callbacks"] = CustomCallbacks

    tune_params = {
        "config": configs,
        "run_or_experiment": conf_yaml["run"],
    }
    if args["tune"]:
        tune_params.update(get_search_alg_sched(conf_yaml, args))
    else:
        tune_params["num_samples"] = 1

    tune_params.update(
        {
            "trial_name_creator": trial_str_creator,
            "trial_dirname_creator": trial_str_creator,
            "stop": {"training_iteration": args["max_iters"]},
            "local_dir": args["logdir"],
            "checkpoint_freq": args["checkpoint_freq"],
            "checkpoint_at_end": args["checkpoint_freq"] > 0,
            "keep_checkpoints_num": 3,
            "checkpoint_score_attr": conf_yaml["metric"],
            "max_failures": 1 if args["smoke"] else 2,
        }
    )

    return tune_params


def get_search_alg_sched(conf_yaml, args):
    alg_name = conf_yaml.get("search_alg")
    metric = conf_yaml["metric"]
    if alg_name is None:
        search_alg = None
    else:
        if alg_name == "hyperopt":
            search_alg = HyperOptSearch(metric=metric, mode="max")
        elif alg_name == "ax":
            search_alg = AxSearch(metric=metric, mode="max")
        else:
            raise NotImplementedError(f"Search alg {alg_name} not implemented")
        search_alg = ConcurrencyLimiter(search_alg, max_concurrent=args["concurrency"])

    if args["no_sched"]:
        scheduler = None
    else:
        scheduler = AsyncHyperBandScheduler(
            time_attr="training_iteration",
            metric=metric,
            mode="max",
            grace_period=50,
            max_t=max(args["max_iters"], 5),
        )

    return {
        "search_alg": search_alg,
        "scheduler": scheduler,
        "num_samples": args["num_samples"],
    }

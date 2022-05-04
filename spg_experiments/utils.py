import json
import time
from datetime import datetime
from os import path as osp

import yaml
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune.registry import ENV_CREATOR, _global_registry, register_env
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.ax import AxSearch
from ray.tune.suggest.hebo import HEBOSearch
from ray.tune.suggest.hyperopt import HyperOptSearch

from .callbacks import CustomCallbacks


class E(dict):
    def keys(self):
        return []


def register():
    # pylint: disable=import-outside-toplevel,self-assigning-variable

    from . import envs, models

    register_env("spg_flat", envs.PgFlat)
    register_env("spg_dict", envs.PgDict)
    register_env("spg_stacked", envs.PgStacked)
    register_env("spg_set", envs.PgSet)
    register_env("spg_graph", envs.PgGraph)

    register_env("atari_raw", envs.AtariRaw)
    register_env("atari_set", envs.AtariSet)
    register_env("atari_graph", envs.AtariGraph)

    ModelCatalog.register_custom_model("fc_net", models.FcPolicy)
    ModelCatalog.register_custom_model("cnn1d_net", models.Cnn1DPolicy)
    ModelCatalog.register_custom_model("attn_net", models.AttnPolicy)
    ModelCatalog.register_custom_model("gnn_net", models.GnnPolicy)


def get_env_creator(env_name):
    return _global_registry.get(ENV_CREATOR, env_name)


def exp_name(prefix):
    return prefix + "." + datetime.now().strftime("%Y-%m-%d.%H:%M:%S")


def trial_str_creator(trial):
    trial_base_name = f"trial-{trial.trial_id}"
    if not trial.evaluated_params or len(trial.evaluated_params) > 3:
        return trial_base_name

    params = {
        k.split("/")[-1]: p[-1] if isinstance(p, list) else str(p)
        for k, p in trial.evaluated_params.items()
    }
    name = "-".join([f"{k}:{p}" for k, p in params.items()])
    name = name.replace("/", "_")
    return f"{trial_base_name}-{name}"


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


def parse_tune_configs(configs, use_tune=False):
    exp = configs["base"]

    is_grid_search = False
    if not use_tune:
        return exp, is_grid_search

    for k, v in configs.get("tune", {}).items():
        if v["type"] == "grid_search":
            is_grid_search = True
            value = getattr(tune, v["type"])(v["args"])

        else:
            if v["type"] == "choice":
                value = getattr(tune, v["type"])(v["args"])

            else:
                value = getattr(tune, v["type"])(*v["args"])

        update_recursive(exp, k, value)

    return exp, is_grid_search


def get_tune_params(args):
    if args["smoke"]:
        args["max_iters"] = 1

    args["num_samples"] = (
        min(2, args["num_samples"]) if args["smoke"] else args["num_samples"]
    )

    cpus_per_worker = (
        args["cpus_per_worker"] / 2 if args["eval_int"] else args["cpus_per_worker"]
    )
    configs_base = {
        "num_workers": args["num_workers"],
        "evaluation_config": {
            "env_config": {"is_eval": True},
        },
        "evaluation_interval": args["eval_int"],
        "evaluation_num_episodes": 10,
        "num_cpus_per_worker": cpus_per_worker,
        "evaluation_num_workers": args["num_workers"] if args["eval_int"] else 0,
        "num_gpus": args["num_gpus"],
        "framework": "torch",
        "num_envs_per_worker": args["envs_per_worker"],
        "batch_mode": "truncate_episodes",
        "observation_filter": "NoFilter",
        "placement_strategy": "SPREAD",
        # "preprocessor_pref": None,
    }

    conf_yaml = get_configs(args["logdir"])
    configs, is_grid_search = parse_tune_configs(conf_yaml, args["tune"])

    # TODO: the same env_config is used for the built-in envs, should be renamed,
    # otherwise they interfere
    if "env_config" not in configs:
        configs["env_config"] = {}

    else:
        # Hack to make serving easier
        configs["env_config"]["__run"] = conf_yaml["run"]
        configs["env_config"]["logdir"] = osp.realpath(args["logdir"])

    configs.update(configs_base)
    configs["callbacks"] = CustomCallbacks

    tune_params = {
        "config": configs,
        "run_or_experiment": conf_yaml["run"],
    }
    tune_params.update(get_search_alg_sched(conf_yaml, args, is_grid_search))

    if args["max_iters"]:
        stop = {"training_iteration": args["max_iters"]}
    elif args["max_steps"]:
        stop = {"timesteps_total": args["max_steps"]}
    else:
        stop = {}

    tune_params.update(
        {
            "trial_name_creator": trial_str_creator,
            "trial_dirname_creator": trial_str_creator,
            "stop": stop,
            "local_dir": args["logdir"],
            "checkpoint_freq": args["checkpoint_freq"],
            "checkpoint_at_end": args["checkpoint_freq"] > 0,
            "keep_checkpoints_num": None if args["keep_all_chkp"] else 3,
            "checkpoint_score_attr": conf_yaml["metric"],
            "max_failures": 1 if args["smoke"] else 2,
        }
    )

    return tune_params


def get_search_alg_sched(conf_yaml, args, is_grid_search):
    if not args["tune"]:
        return {"num_samples": args["num_samples"]}

    alg_name = conf_yaml.get("search_alg")
    metric = conf_yaml["metric"]

    if alg_name is None or is_grid_search:
        search_alg = None

    else:
        assert args["num_samples"] > 1

        if alg_name == "hyperopt":
            search_alg = HyperOptSearch(metric=metric, mode="max", n_initial_points=10)
        elif alg_name == "ax":
            search_alg = AxSearch(metric=metric, mode="max")
        elif alg_name == "hebo":
            search_alg = HEBOSearch(metric=metric, mode="max")
        else:
            raise NotImplementedError(f"Search alg {alg_name} not implemented")

        search_alg = ConcurrencyLimiter(search_alg, max_concurrent=args["concurrency"])

    if args["no_sched"] or args["smoke"]:
        scheduler = None

    else:
        scheduler = AsyncHyperBandScheduler(
            time_attr="training_iteration",
            metric=metric,
            mode="max",
            grace_period=args["grace_period"],
            max_t=max(args["max_iters"], 5),
        )

    return {
        "search_alg": search_alg,
        "scheduler": scheduler,
        "num_samples": args["num_samples"],
    }


class TicToc:
    def __init__(self):
        self.start = 0
        self.lap = 0

        self.tic()

    def tic(self):
        self.start = time.time()
        self.lap = self.start

    def toc(self, message=""):
        now = time.time()
        elapsed_1 = now - self.start
        elapsed_2 = now - self.lap
        m = f"Cum: {elapsed_1:.6f}\tLap: {elapsed_2:.6f}"
        if message:
            m += f", {message}"
        # logging.debug(m)
        print(m)
        self.lap = now

    def cum(self):
        now = time.time()
        elapsed = now - self.start
        m = f"Cum: {elapsed:.6f}\t"
        print(m)
        # logging.debug(m)

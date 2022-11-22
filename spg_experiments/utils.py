import json
import time
from datetime import datetime
from functools import wraps
from os import path as osp

import yaml
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune.registry import ENV_CREATOR, _global_registry, register_env
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.ax import AxSearch
from ray.tune.search.hebo import HEBOSearch
from ray.tune.search.hyperopt import HyperOptSearch

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
    register_env("corridor", envs.atari.base.SimpleCorridor)

    ModelCatalog.register_custom_model("fc_net", models.FcPolicy)
    ModelCatalog.register_custom_model("cnn1d_net", models.Cnn1DPolicy)
    ModelCatalog.register_custom_model("cnn_net", models.CnnPolicy)
    ModelCatalog.register_custom_model("attn_net", models.AttnPolicy)
    ModelCatalog.register_custom_model("gnn_net", models.GnnPolicy)
    # ModelCatalog.register_custom_model("space_gnn_net", models.SpaceGnnPolicy)


def get_env_creator(env_name):
    return _global_registry.get(ENV_CREATOR, env_name)


def exp_name(prefix):
    if not isinstance(prefix, str):
        prefix = prefix.__name__

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
        args["stop_attr"] = "training_iteration"
        args["stop_at"] = 1

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
        "evaluation_duration": 10,
        "evaluation_duration_unit": "episodes",
        "num_cpus_per_worker": cpus_per_worker,
        "evaluation_num_workers": args["num_workers"] if args["eval_int"] else 0,
        "num_gpus": args["num_gpus"],
        "framework": "torch",
        "num_envs_per_worker": args["envs_per_worker"],
        "batch_mode": "truncate_episodes",
        "observation_filter": "NoFilter",
        "log_sys_usage": False,
        # "preprocessor_pref": None,
    }

    if args["no_gpu_workers"]:
        configs_base["custom_resources_per_worker"] = {"NO-GPU": 0.0001}

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

    assert conf_yaml["run"] == "PPO"
    if "space_loss_coeff" in configs:
        raise NotImplementedError("SpacePPOTrainer is not fully implemented.")

    experiment = "PPO"

    tune_params = {
        "config": configs,
        "run_or_experiment": experiment,
    }
    tune_params.update(get_search_alg_sched(conf_yaml, args, is_grid_search))

    tune_params.update(
        {
            "trial_name_creator": trial_str_creator,
            "trial_dirname_creator": trial_str_creator,
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
    stop = {args["stop_attr"]: args["stop_at"]}

    if not args["tune"]:
        return {"stop": stop, "num_samples": args["num_samples"]}

    alg_name = conf_yaml.get("search_alg")
    metric = conf_yaml["metric"]

    if alg_name is None or is_grid_search:
        # FIXME: concurrency limiter doesn't work this way
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
        assert 0 < args["grace_period"] < 1
        scheduler = AsyncHyperBandScheduler(
            time_attr=args["stop_attr"],
            metric=metric,
            mode="max",
            grace_period=int(args["stop_at"] * args["grace_period"]),
            max_t=args["stop_at"],
        )

    return {
        "stop": stop,
        "search_alg": search_alg,
        "scheduler": scheduler,
        "num_samples": args["num_samples"],
    }


class TicToc:
    def __init__(self):
        self.start = 0
        self.lap = 0
        self.counter = 0

        self.tic()

    def tic(self):
        self.start = time.time()
        self.lap = self.start
        self.counter = 0

    def toc(self, message=""):
        now = time.time()
        elapsed_1 = now - self.start
        elapsed_2 = now - self.lap
        m = f"{self.counter}\tCum: {elapsed_1:.6f}\tLap: {elapsed_2:.6f}"
        if message:
            m += f", {message}"
        # logging.debug(m)
        print(m)
        self.lap = now
        self.counter += 1

    def cum(self):
        now = time.time()
        elapsed = now - self.start
        m = f"Cum: {elapsed:.6f}\t"
        print(m)
        # logging.debug(m)


def timing_wrapper(method):
    @wraps(method)
    def wrapped(*args, **kwargs):
        print("#####")
        print("Calling", method.__name__)
        start_time = time.time()
        result = method(*args, **kwargs)
        print("Done", method.__name__, "Took", time.time() - start_time)
        return result

    return wrapped


def wrap_methods(cls, wrapper):
    for key, value in cls.__dict__.items():
        if hasattr(value, "__call__"):
            setattr(cls, key, wrapper(value))

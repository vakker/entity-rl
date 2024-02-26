import argparse
import logging
import os
from os import path as osp

import ray
import torch
from ray.tune.tuner import Tuner

from entity_rl import utils

torch.backends.cudnn.benchmark = True


def main(args):
    args_dict = vars(args)
    os.environ["TUNE_RESULT_DIR"] = args_dict["logdir"]

    ray.init(
        address=args_dict["head_ip"],
        _node_ip_address=args_dict["node_ip"],
        local_mode=args_dict["local"],
        configure_logging=True,
        logging_level="DEBUG" if args.verbose else "INFO",
    )
    utils.register()

    configs = utils.get_configs(args_dict)

    if args_dict["resume_from"]:
        tuner = Tuner.restore(
            path=args_dict["resume_from"],
            trainable=configs["trainable"],
        )

    else:
        tuner = Tuner(**configs)

    results = tuner.fit()
    return results


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("-v", "--verbose", action="store_true")
    PARSER.add_argument("-s", "--smoke", action="store_true")
    PARSER.add_argument("--resume-from")
    PARSER.add_argument("--errored-only", action="store_true")
    PARSER.add_argument("--tune", action="store_true")
    PARSER.add_argument("--no-sched", action="store_true")
    PARSER.add_argument("--logdir")
    PARSER.add_argument("--stop-attr", type=str, default="timesteps_total")
    PARSER.add_argument("--stop-at", type=int, default=100000)
    PARSER.add_argument("--num-samples", type=int, default=1)
    PARSER.add_argument("--local", action="store_true")
    PARSER.add_argument("--concurrency", type=int, default=100000)
    PARSER.add_argument("--num-workers", type=int, default=1)
    PARSER.add_argument("--cpus-per-worker", type=float, default=1.0)
    PARSER.add_argument("--gpus-per-worker", type=float, default=0.0)
    PARSER.add_argument("--envs-per-worker", type=int, default=1)
    PARSER.add_argument("--grace-period", type=float, default=0.25)
    PARSER.add_argument("--num-gpus", type=float, default=1.0)
    PARSER.add_argument("--checkpoint-freq", type=int, default=1)
    PARSER.add_argument("--keep-all-chkp", action="store_true")
    PARSER.add_argument("--eval-int", type=int)
    PARSER.add_argument("--node-ip", type=str, default="127.0.0.1")
    PARSER.add_argument("--head-ip", type=str)
    PARSER.add_argument("--num-cpus", type=str)
    PARSER.add_argument("--exp-name", type=str, default="SPG-EXP")
    PARSER.add_argument("--no-gpu-workers", action="store_true")
    PARSER.add_argument("--callbacks", nargs="+")

    ARGS = PARSER.parse_args()

    loglevel = "DEBUG" if ARGS.verbose else "INFO"
    level = logging.getLevelName(loglevel)
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=level,
        format=log_format,
        filename=osp.join(ARGS.logdir, "train.log"),
        filemode="a",
    )

    main(ARGS)

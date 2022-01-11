import argparse
import logging
from os import path as osp

import ray
import torch
from ray import tune
from ray.tune import CLIReporter

from spg_experiments import callbacks, utils

torch.backends.cudnn.benchmark = True


def main(args):
    args_dict = vars(args)

    assert (args.max_iters is None) ^ (
        args.max_steps is None
    ), "Provide either max iters or steps."

    ray.init(
        local_mode=args_dict["local"],
        configure_logging=True,
        logging_level="DEBUG" if args.verbose else "INFO",
    )
    utils.register()

    if args_dict["local"] and args_dict["num_workers"]:
        args_dict["num_workers"] = 1

    tune_params = utils.get_tune_params(args_dict)
    if args_dict["resume_from"]:
        name = args_dict["resume_from"]
        resume = True
    else:
        name = utils.exp_name(tune_params["run_or_experiment"])
        resume = False

    reporter = CLIReporter(parameter_columns=utils.E({"_": "_"}))
    analysis = tune.run(
        **tune_params,
        progress_reporter=reporter,
        resume=resume,
        name=name,
        verbose=3 if args_dict["verbose"] else 1,
        callbacks=[callbacks.DataLoggerCallback()]
    )

    return analysis


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("-v", "--verbose", action="store_true")
    PARSER.add_argument("-s", "--smoke", action="store_true")
    PARSER.add_argument("--resume-from")
    PARSER.add_argument("--errored-only", action="store_true")
    PARSER.add_argument("--tune", action="store_true")
    PARSER.add_argument("--no-sched", action="store_true")
    PARSER.add_argument("--logdir")
    PARSER.add_argument("--max-iters", type=int)
    PARSER.add_argument("--max-steps", type=int)
    PARSER.add_argument("--num-samples", type=int, default=1)
    PARSER.add_argument("--local", action="store_true")
    PARSER.add_argument("--concurrency", type=int, default=1)
    PARSER.add_argument("--num-workers", type=int, default=1)
    PARSER.add_argument("--cpus-per-worker", type=float, default=1.0)
    PARSER.add_argument("--envs-per-worker", type=int, default=1)
    PARSER.add_argument("--num-gpus", type=float, default=1.0)
    PARSER.add_argument("--checkpoint-freq", type=int, default=1)
    PARSER.add_argument("--eval-int", type=int)

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

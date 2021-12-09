from ray.rllib.rollout import *
from ray.tune.registry import register_env

from spg_experiments import models
from spg_experiments.gym_env import PlaygroundEnv

register_env("spg-v0", lambda config: PlaygroundEnv(config))


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    # --use_shelve w/o --out option.
    if args.use_shelve and not args.out:
        raise ValueError(
            "If you set --use-shelve, you must provide an output file via "
            "--out as well!"
        )
    # --track-progress w/o --out option.
    if args.track_progress and not args.out:
        raise ValueError(
            "If you set --track-progress, you must provide an output file via "
            "--out as well!"
        )

    run(args, parser)

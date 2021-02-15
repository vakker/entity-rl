import argparse

import yaml

from spg_experiments.wrappers.train_eval import train_and_eval


def main(args):
    # eval params

    fname = (
        args.environment
        + "_"
        + args.sensors
        + "_"
        + str(args.entropy)
        + "_"
        + str(args.multistep)
        + "_"
        + str(args.index_exp)
    )

    agent_type = "base"

    results = train_and_eval(
        agent_type,
        args.sensors,
        total_timesteps_training=int(1e6),
        n_multisteps=args.multistep,
        playground_name=args.environment,
        freq_eval=int(1e5),
        episodes_eval=5,
        entropy=args.entropy,
    )

    log_fname = "logs/" + fname + ".dat"

    with open(log_fname, "w") as f:
        yaml.dump(results, f)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--environment")
    PARSER.add_argument("--sensors")
    PARSER.add_argument("--entropy", type=float)
    PARSER.add_argument("--multistep", type=int)
    PARSER.add_argument("--index-exp", type=int)

    ARGS = PARSER.parse_args()

    main(ARGS)

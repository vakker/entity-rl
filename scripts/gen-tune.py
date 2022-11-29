import argparse

import yaml


def main(args):
    with open(args.games) as yml_file:
        games = yaml.safe_load(yml_file)

    games = list(sorted(games.items(), key=lambda x: x[1]))
    games = [g[0] for g in games if g[1] < args.max_x]
    tune_dict = {"tune": {"env_config/pg_name": {"type": "grid_search", "args": games}}}

    with open("sorted-envs.yaml", "w") as yml_file:
        yaml.safe_dump(tune_dict, yml_file)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--games")
    PARSER.add_argument("--max-x", type=int, default=60)

    ARGS = PARSER.parse_args()

    main(ARGS)

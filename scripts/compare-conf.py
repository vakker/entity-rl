import argparse

import yaml
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.utils import flatten_dict


def main(args):
    config1 = PPOConfig.from_dict(yaml.safe_load(open(args.config1))["base"]).to_dict()
    config2 = PPOConfig.from_dict(yaml.safe_load(open(args.config2))["base"]).to_dict()

    config1 = flatten_dict(config1)
    config2 = flatten_dict(config2)

    elements_1 = sorted(list(config1.keys()))
    elements_2 = sorted(list(config2.keys()))

    keys = set(elements_1) | set(elements_2)

    for element in sorted(keys):
        value1 = config1.get(element)
        value2 = config2.get(element)
        if value1 != value2:
            print(f"!! {element}")
            print(f"Config 1: {value1}")
            print(f"Config 2: {value2}")
            print()

        elif args.print:
            print(f"== {element}")
            print(value1)
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config1", help="Config file")
    parser.add_argument("config2", help="Config file")
    parser.add_argument("-p", "--print", action="store_true")
    args = parser.parse_args()
    main(args)

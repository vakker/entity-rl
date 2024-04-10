import argparse

import yaml
from ray.rllib.algorithms.ppo import PPOConfig


def main(args):
    config1 = PPOConfig.from_dict(yaml.safe_load(open(args.config1))["base"]).to_dict()
    config2 = PPOConfig.from_dict(yaml.safe_load(open(args.config2))["base"]).to_dict()

    elements_1 = sorted(list(config1.keys()))
    elements_2 = sorted(list(config2.keys()))

    assert set(elements_1) == set(elements_2)

    for element in elements_1:
        value1 = config1[element]
        value2 = config2[element]
        if value1 != value2:
            print(f"Element {element} differs")
            print(f"Config 1: {value1}")
            print(f"Config 2: {value2}")
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config1", help="Config file")
    parser.add_argument("config2", help="Config file")
    args = parser.parse_args()
    main(args)

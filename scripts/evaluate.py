from ray.rllib import evaluate

from entity_rl import utils

if __name__ == "__main__":
    utils.register()

    evaluate.main()

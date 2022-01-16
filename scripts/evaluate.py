from ray.rllib import evaluate

from spg_experiments import utils

if __name__ == "__main__":
    utils.register()

    evaluate.main()

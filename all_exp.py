from argparse import Namespace

from spg_exp import main

args = Namespace()
if __name__ == "__main__":
    for index_exp in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        for environment in [
                "spg_endgoal_cue",
                "spg_endgoal_9rooms",
                "spg_dispenser_6rooms",
                "spg_coinmaster_singleroom",
        ]:
            for sensors in ["rgb", "rgb_depth rgb_touch", "rgb_depth_touch"]:
                for entropy in [0.05, 0.01, 0.005, 0.001]:
                    for multistep in [0, 2, 3, 4]:
                        args = Namespace(
                            index_exp=index_exp,
                            environment=environment,
                            sensors=sensors,
                            entropy=entropy,
                            multistep=multistep,
                        )
                        main(args)

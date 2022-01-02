import argparse
import os
import shutil
from os import path as osp

import matplotlib.pyplot as plt
import numpy as np
import requests
from skimage import io as skio
from tqdm import trange

from spg_experiments import utils


def get_action(obs, port):
    if isinstance(obs, dict):
        obs = {k: v.tolist() for k, v in obs.items()}
    else:
        obs = obs.tolist()

    resp = requests.get(f"http://localhost:{port}/agent", json={"obs": obs})
    return resp.json()["action"]


def main(args):
    utils.register()
    env_creator = utils.get_env_creator(args.env)
    env = env_creator({"pg_name": args.pg, "sensors_name": "rgb_depth_touch"})

    if args.save:
        if osp.exists("frames"):
            print("Frames dir exists, removing")
            shutil.rmtree("frames")

        os.mkdir("frames")

    obs = env.reset()

    plot_obj = None

    for j in trange(args.iters, disable=args.no_bar):
        if args.use_serve:
            act = get_action(obs, args.serve_port)
        else:
            act = np.random.rand(2, 1) * 2 - 1

        obs, _, done, _ = env.step(act)

        if args.save:
            skio.imsave(f"frames/frame-{j:06d}.png", env.full_scenario())

        if args.render:
            if plot_obj is None:
                plot_obj = plt.imshow(env.full_scenario())
                plt.show(block=False)

            plot_obj.set_data(env.full_scenario())
            plt.gcf().canvas.draw_idle()
            plt.gcf().canvas.start_event_loop(0.01)

        if done:
            break

    env.reset()


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--pg", default="basic_rl/candy_poison")
    PARSER.add_argument("--env", default="spg_dict")
    PARSER.add_argument("--no-bar", action="store_true")
    PARSER.add_argument("-r", "--render", action="store_true")
    PARSER.add_argument("-s", "--save", action="store_true")
    PARSER.add_argument("-k", "--keys", action="store_true")
    PARSER.add_argument("-d", "--debug", action="store_true")
    PARSER.add_argument("-i", "--iters", type=int, default=100)
    PARSER.add_argument("--use-serve", action="store_true")
    PARSER.add_argument("--serve-port", type=int, default=7999)

    ARGS = PARSER.parse_args()

    main(ARGS)

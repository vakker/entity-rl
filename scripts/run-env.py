# pylint: disable=too-many-branches,all
import argparse
import os
import pickle
import shutil
from functools import partial
from multiprocessing import Pool
from os import path as osp

import numpy as np
import requests
from skimage import io as skio
from tqdm import tqdm, trange

from entity_rl import utils


def get_action(obs, port):
    if isinstance(obs, dict):
        obs = {k: v.tolist() for k, v in obs.items()}
    else:
        obs = obs.tolist()

    resp = requests.get(f"http://localhost:{port}/agent", json={"obs": obs})
    return resp.json()["action"]


def run_episode(conf_yaml, args, frames_base, i):
    env_creator = utils.get_env_creator(conf_yaml["env"])
    if args.pg_name:
        conf_yaml["env_config"]["pg_name"] = args.pg_name

    env = env_creator(conf_yaml["env_config"])
    env.render_mode = "rgb_array"
    env.is_eval = True

    frames_dir = osp.join(frames_base, f"episode-{i:02d}")
    obs, _ = env.reset()

    dataset = []
    for j in range(args.iters):
        if args.use_serve:
            act = get_action(obs, args.serve_port)

        elif args.keys:
            act = env.agent.controller.generate_actions()
            act = [v for k, v in act.items()]

        else:
            act = env.action_space.sample()
            # act[0] = 2

        obs, reward, done, _, _ = env.step(act)

        if args.save:
            dataset.append({"obs": obs, "act": act, "reward": reward})

        if args.render:
            if not osp.exists(frames_dir):
                os.mkdir(frames_dir)

            img = env.render()
            assert img.shape[-1] == 3

            img_path = osp.join(frames_dir, f"f-{j:06d}.png")
            skio.imsave(img_path, img, check_contrast=False)

        # if args.render:
        #     if plot_obj is None:
        #         img = env.render()
        #         assert img.shape[-1] == 3

        #         plot_obj = plt.imshow(img)
        #         plt.show(block=False)

        #     plot_obj.set_data(env.full_scenario())
        #     plt.gcf().canvas.draw_idle()
        #     plt.gcf().canvas.start_event_loop(0.01)

        if done:
            break

    return dataset


def main(args):
    if args.conf:
        conf_yaml = utils.load_dict(args.conf)["base"]

    utils.register()

    # env = env_creator(
    #     {
    #         "pg_name": args.pg,
    #         "sensors_name": "rgb_depth_touch",
    #         "keyboard": args.keys,
    #     }
    # )

    if args.postfix:
        frames_base = f"frames-{args.postfix}"
    else:
        frames_base = "frames"

    if args.output_dir:
        frames_base = osp.join(args.output_dir, frames_base)

    if args.save or args.render:
        print(f"Saving frames to {frames_base}")
        if osp.exists(frames_base):
            print("Frames dir exists, removing")
            shutil.rmtree(frames_base)

        os.mkdir(frames_base)

    # plot_obj = None
    dataset = []
    f = partial(run_episode, conf_yaml, args, frames_base)

    if args.num_p > 0:
        with Pool(processes=args.num_p) as pool:
            it = pool.imap_unordered(f, range(args.episodes))
            for d in tqdm(it, total=args.episodes):
                dataset.extend(d)
    else:
        for i in trange(args.episodes, disable=args.no_bar):
            dataset.extend(f(i))

    if args.save:
        print("Saving dataset")
        if args.no_filter:
            data_filtered = dataset

        else:
            data_filtered = [d for d in dataset if d["reward"] != 0]
            zero_rewards = [d for d in dataset if d["reward"] == 0]
            data_filtered.extend(np.random.choice(zero_rewards, len(data_filtered)))

        rewards = [d["reward"] for d in data_filtered]
        print("Data stats:")
        print(f"Total: {len(data_filtered)}")
        values, counts = np.unique(rewards, return_counts=True)
        for v, n in zip(values, counts):
            print(f"Reward {v}: {n}")

        pkl_path = osp.join(frames_base, "dataset.pkl")
        with open(pkl_path, "wb") as file:
            pickle.dump(data_filtered, file)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--pg-name")
    PARSER.add_argument("--env")
    PARSER.add_argument("--conf")
    PARSER.add_argument("--output-dir")
    PARSER.add_argument("--no-bar", action="store_true")
    PARSER.add_argument("-r", "--render", action="store_true")
    PARSER.add_argument("-s", "--save", action="store_true")
    PARSER.add_argument("-k", "--keys", action="store_true")
    PARSER.add_argument("-d", "--debug", action="store_true")
    PARSER.add_argument("--postfix", type=str)
    PARSER.add_argument("-i", "--iters", type=int, default=100)
    PARSER.add_argument("-e", "--episodes", type=int, default=1)
    PARSER.add_argument("--use-serve", action="store_true")
    PARSER.add_argument("--no-filter", action="store_true")
    PARSER.add_argument("--serve-port", type=int, default=7999)
    PARSER.add_argument("--num-p", type=int, default=10)

    ARGS = PARSER.parse_args()

    main(ARGS)

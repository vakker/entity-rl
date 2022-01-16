# Experiments for "Efficient entity-based reinforcement learning"

This repository contains all the code and configs used to produce the results
for "Efficient entity-based reinforcement learning" submitted to IJCAI22.

## Setup

Install the requirements with `pip install -r requirements.txt`, make sure that
the Cuda version for PyG matches your installed version.

Then run `pip install <-e> .`.

## Experiments

Each experiment is defined by a config file, for the tuned examples see the
`configs` directory.

Create a log directory and place a config file in it named `conf.yaml`. Then run

``` sh
python scripts/train.py --logdir <logdir> --num-workers <num-workers> --max-iters <max-iters>
```

Use `--tune --no-sched` to go through each option under the `tune` section in
the config file. For SPG ~200-400 iterations is enough, for Atari it's around
2000-4000.

Note, that Atari was only tuned on Pong, but it can be run on any games
supported by ALE.

For videos of the trained agents performing see [this
playlist](https://www.youtube.com/playlist?list=PL0fzH_bs_m9jy4uzf8Oj5TP11OVEPxATh).

If you have any issues, please reach out on: entity.rl.ijcai22@gmail.com

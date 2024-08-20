# Experiments for "ENROS: Entity-based Reinforcement Learning with Open-world Scene Understanding"

This repository contains all the code and configs used to produce the results
for ENROS.

## Setup

Install the requirements with `pip install -r requirements.txt`, make sure that
the Cuda version for PyG matches your installed version.

Then run `pip install <-e> .`.

## Experiments 1

Each experiment is defined by a config file, for the tuned examples see the
`configs` directory.

Create a log directory and place a config file in it named `conf.yaml`. Then run

```sh
python scripts/train.py --logdir <logdir>
```

See example configs in the `configs` directory.

Note, that Atari was only tuned on Pong, but it can be run on any games
supported by ALE.

For videos of the trained agents performing see [this
playlist](https://www.youtube.com/playlist?list=PL0fzH_bs_m9jy4uzf8Oj5TP11OVEPxATh).


## Experiments 2

The proxy task use `scripts/train-gdino.py` with the same config file format
as the previous experiments.

The experiences are generated with `scripts/run-env.py`, and passed to the script:

```sh
python scripts/train-gdino.py --cfg <configfile> --data <datafile>
```


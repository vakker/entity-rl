#!/usr/bin/env bash

for index_exp in 0 1 2 3 4 5 6 7 8 9; do
	for environment in spg_endgoal_cue spg_endgoal_9rooms spg_dispenser_6rooms spg_coinmaster_singleroom; do
		for sensor in rgb rgb_depth rgb_touch rgb_depth_touch; do
			for entropy in 0.05 0.01 0.005 0.001; do
				for multistep in 0 2 3 4; do
					python3 spg_exp.py $environment $sensor $entropy $multistep $index_exp
				done
			done
		done
	done
done

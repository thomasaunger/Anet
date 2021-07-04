#!/usr/bin/env python3

"""
Script to test a sender and receiver through reinforcement learning.
"""

import gym
import time
import datetime
import torch
import numpy as np

import anet.utils as utils

import babyai.utils

from anet.arguments         import ArgumentParser
from anet.rl.algos          import BaseAlgo
from anet.tasks.mnist.utils import ParallelEnv

# Parse arguments.
parser = ArgumentParser()

# Model parameters
parser.add_argument("--sender", default=None,
                    help="name of the sender (REQUIRED)")
parser.add_argument("--receiver", default=None,
                    help="name of the receiver (REQUIRED)")

# Environment arguments
parser.add_argument("--archimedean", action="store_true", default=False,
                    help="use Archimedean receiver")

# Algorithm arguments
parser.add_argument("--frames-per-proc", type=int, default=160,
                    help="number of frames per process before update (default: 160)")

# Testing arguments
parser.add_argument("--sample", action="store_true", default=False,
                    help="sample messages instead of using argmax")

args = parser.parse_args()

babyai.utils.seed(args.seed)

# Generate environments.
envs = []
for i in range(args.procs):
    env = gym.make(args.env, procs=args.procs, proc_id=i)
    env.seed(args.seed)
    envs.append(env)

penv = ParallelEnv(envs, args.conventional, args.archimedean)

# Define obss preprocessor.
model_names = [args.sender, args.receiver]
obss_preprocessor = utils.MultiObssPreprocessor(model_names, [envs[0].observation_space for _ in model_names], model_names)

# Define actor--critic models.
sender   = babyai.utils.load_model(args.sender)
receiver = babyai.utils.load_model(args.receiver)

if torch.cuda.is_available():
    sender.cuda()
    receiver.cuda()

# Define actor--critic algorithm.
reshape_reward = lambda _0, _1, reward, _2: args.reward_scale * reward
base_algo = BaseAlgo(penv, [sender, receiver], args.frames_per_proc, args.discount, args.gae_lambda, obss_preprocessor, reshape_reward, not args.no_comm, args.conventional, not args.sample, args.single_precision)

# Test models.
sender.eval()
receiver.eval()
total_start_time = time.time()
update_start_time = time.time()
_, logs = base_algo.collect_experiences()
update_end_time = time.time()

SENDER   = 0
RECEIVER = 1

# Print log.
total_ellapsed_time = int(time.time() - total_start_time)
fps = logs[RECEIVER]["num_frames"] / (update_end_time - update_start_time)
duration = datetime.timedelta(seconds=total_ellapsed_time)
return_per_episode = babyai.utils.synthesize(logs[RECEIVER]["return_per_episode"])
success_per_episode = babyai.utils.synthesize(
    [1 if r > 0 else 0 for r in logs[RECEIVER]["return_per_episode"]])
num_frames_per_episode = babyai.utils.synthesize(logs[RECEIVER]["num_frames_per_episode"])

data = [logs[RECEIVER]["episodes_done"], logs[RECEIVER]["num_frames"],
        fps, total_ellapsed_time,
        *return_per_episode.values(),
        success_per_episode["mean"],
        *num_frames_per_episode.values()]

format_str = ("E {} | F {:06} | FPS {:04.0f} | D {} | R:xsmM {: .2f} {: .2f} {: .2f} {: .2f} | "
              "S {:.2f} | F:xsmM {:.1f} {:.1f} {} {} | ")

print(format_str.format(*data))

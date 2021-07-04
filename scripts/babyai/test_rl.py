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

from anet.arguments          import ArgumentParser
from anet.rl.algos           import BaseAlgo
from anet.tasks.babyai.utils import ParallelEnv

def collect_episodes(base_algo, episodes):
    # Collect experiences.
    exps, _ = base_algo.collect_experiences()
    batch   = 1
    
    active  = exps.active.view( base_algo.num_procs, base_algo.num_frames_per_proc, *exps.active.shape[ 1:])
    extra   = exps.extra.view(  base_algo.num_procs, base_algo.num_frames_per_proc, *exps.extra.shape[  1:])
    mask    = exps.mask.view(   base_algo.num_procs, base_algo.num_frames_per_proc, *exps.mask.shape[   1:])
    done    = exps.done.view(   base_algo.num_procs, base_algo.num_frames_per_proc, *exps.done.shape[   1:])
    message = exps.message.view(base_algo.num_procs, base_algo.num_frames_per_proc, *exps.message.shape[1:])
    action  = exps.action.view( base_algo.num_procs, base_algo.num_frames_per_proc, *exps.action.shape[ 1:])
    reward  = exps.reward.view( base_algo.num_procs, base_algo.num_frames_per_proc, *exps.reward.shape[ 1:])
    
    log = {
        "return_per_episode":     [],
        "num_frames_per_episode": [],
        "num_frames":             0,
        "episodes_done":          0,
    }
    
    exp_data = torch.zeros(base_algo.num_frames, 4 + base_algo.models[0].len_msg + 3 + 5, dtype=torch.uint8)
    
    SENDER   = 0
    RECEIVER = 1
    
    t             = 0
    proc          = 0
    frame         = [0]*base_algo.num_procs
    episode_frame = 0
    episodes_done = 0
    while True:
        if active[proc, frame[proc], RECEIVER]:
            exp_data[t,                          0:4                         ] = extra[  proc, frame[proc],        0:4]
            exp_data[t,                          4:4+base_algo.models[0].len_msg  ] = message[proc, frame[proc], SENDER  ].argmax(-1)
            exp_data[t, 4+base_algo.models[0].len_msg  ]                            = action[ proc, frame[proc], RECEIVER]
            exp_data[t, 4+base_algo.models[0].len_msg+1]                            = ~mask[  proc, frame[proc], RECEIVER]
            exp_data[t, 4+base_algo.models[0].len_msg+2]                            = reward[ proc, frame[proc], RECEIVER].ceil()
            exp_data[t, 4+base_algo.models[0].len_msg+3:4+base_algo.models[0].len_msg+5] = extra[  proc, frame[proc],        4:6]
            exp_data[t, 4+base_algo.models[0].len_msg+5:4+base_algo.models[0].len_msg+7] = extra[  proc, frame[proc],        2:4]
            
            t             += 1
            episode_frame += 1
        
        if done[proc, frame[proc]]:
            episodes_done += 1
            log["return_per_episode"].append(reward[proc, frame[proc], RECEIVER].item())
            log["num_frames_per_episode"].append(episode_frame)
            episode_frame = 0
            if episodes_done == episodes:
                break
            frame[proc] += 1
            proc         = (proc + 1) % base_algo.num_procs
        else:
            frame[proc] += 1
        
        if t == exp_data.shape[0]:
            exp_data = torch.cat((exp_data, torch.zeros(exp_data.shape, dtype=torch.uint8)), 0)
        
        if frame[proc] == batch*base_algo.num_frames_per_proc:
            exps, _  = base_algo.collect_experiences()
            batch   += 1
            
            next_active  = exps.active.view( base_algo.num_procs, base_algo.num_frames_per_proc, *exps.active.shape[ 1:])
            next_extra   = exps.extra.view(  base_algo.num_procs, base_algo.num_frames_per_proc, *exps.extra.shape[  1:])
            next_mask    = exps.mask.view(   base_algo.num_procs, base_algo.num_frames_per_proc, *exps.mask.shape[   1:])
            next_done    = exps.done.view(   base_algo.num_procs, base_algo.num_frames_per_proc, *exps.done.shape[   1:])
            next_message = exps.message.view(base_algo.num_procs, base_algo.num_frames_per_proc, *exps.message.shape[1:])
            next_action  = exps.action.view( base_algo.num_procs, base_algo.num_frames_per_proc, *exps.action.shape[ 1:])
            next_reward  = exps.reward.view( base_algo.num_procs, base_algo.num_frames_per_proc, *exps.reward.shape[ 1:])
            
            active  = torch.cat((active,  next_active ), 1)
            extra   = torch.cat((extra,   next_extra  ), 1)
            mask    = torch.cat((mask,    next_mask   ), 1)
            done    = torch.cat((done,    next_done   ), 1)
            message = torch.cat((message, next_message), 1)
            action  = torch.cat((action,  next_action ), 1)
            reward  = torch.cat((reward,  next_reward ), 1)
    
    log["num_frames"]    = t
    log["episodes_done"] = episodes_done
    
    return exp_data[:t], log


# Parse arguments.
parser = ArgumentParser()

# Model parameters
parser.add_argument("--sender", default=None,
                    help="name of the sender (REQUIRED)")
parser.add_argument("--receiver", default=None,
                    help="name of the receiver (REQUIRED)")

# Environment arguments
parser.add_argument("--n", type=int, default=64,
                    help="communication interval (default: 64)")
parser.add_argument("--archimedean", action="store_true", default=False,
                    help="use Archimedean receiver")
parser.add_argument("--informed-sender", action="store_true", default=False,
                    help="allows sender to see the instruction")

# Algorithm arguments
parser.add_argument("--frames-per-proc", type=int, default=160,
                    help="number of frames per process before update (default: 160)")

# Testing arguments
parser.add_argument("--sample", action="store_true", default=False,
                    help="sample messages instead of using argmax")
parser.add_argument("--episodes", type=int, default=1000,
                    help="number of episodes to test on (default: 1000)")

args = parser.parse_args()

babyai.utils.seed(args.seed)

envs = []
for i in range(args.procs):
    env = gym.make(args.env)
    env.seed(100 * args.seed + i)
    envs.append(env)

penv = ParallelEnv(envs, args.n, args.conventional, args.archimedean, args.informed_sender)

# Define obss preprocessor.
model_names = [args.sender, args.receiver]
obss_preprocessor = utils.MultiObssPreprocessor(model_names, [envs[0].observation_space]*2, model_names)

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
exp_data, logs = collect_episodes(base_algo, args.episodes)
update_end_time = time.time()

# Print log.
total_ellapsed_time = int(time.time() - total_start_time)
fps = logs["num_frames"] / (update_end_time - update_start_time)
duration = datetime.timedelta(seconds=total_ellapsed_time)
return_per_episode = babyai.utils.synthesize(logs["return_per_episode"])
success_per_episode = babyai.utils.synthesize(
    [1 if r > 0 else 0 for r in logs["return_per_episode"]])
num_frames_per_episode = babyai.utils.synthesize(logs["num_frames_per_episode"])

data = [logs["episodes_done"], logs["num_frames"],
        fps, total_ellapsed_time,
        *return_per_episode.values(),
        success_per_episode["mean"],
        *num_frames_per_episode.values()]

format_str = ("E {} | F {:06} | FPS {:04.0f} | D {} | R:xsmM {: .2f} {: .2f} {: .2f} {: .2f} | "
              "S {:.2f} | F:xsmM {:.1f} {:.1f} {} {} | ")

np.save("data.npy", exp_data.numpy())

print(format_str.format(*data))

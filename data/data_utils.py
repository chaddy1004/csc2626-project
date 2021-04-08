import glob
import os
import random
import sys

import gym
import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler

sys.path.append(os.path.abspath(os.path.join('..')))
from config import Config as cfg


def rollout_viz(env, data, render=False, seed=None, overfit=False):
    """Visualize previously saved rollouts in the gym env"""

    total_reward = 0
    steps = 0
    s = env.reset()
    if overfit:
        env.seed(seed)
        env.action_space.seed(seed)
    N = len(data)
    for i in range(N):
        s = data[i][:8]
        a = data[i][8:10]
        r = data[i][10]
        ns = data[i][11:19]
        done = data[i][19]
        # e = data[i][20]
        env.step(a)

        total_reward += r

        if render:
            still_open = env.render()
            if still_open == False: break

        if steps % 20 == 0 or done:
            print("observations:", " ".join(["{:+0.2f}".format(x) for x in s]))
            print("step {} total_reward {:+0.2f}".format(steps, total_reward))
        steps += 1
    env.close()
    return total_reward


def get_weighted_sampler(ratio, normalize=False):
    """
    Returns dataframe and WeightedRandomSampler based on desired ratio of sub-optimal to 
    optimal data rollouts
        ratio: split tuple of opt to sub-opt data, e.g. (0.1, 0.9) for
               10% optimal and 90% suboptimal training data
        normalize: whether the weights should be normalized based on count
    """

    assert np.sum(ratio) == 1.0, "Ratios should some up to 1.0"
    ratio *= np.random.uniform(0.999, 1.001, 2)

    files = glob.glob(os.path.join(cfg.DATA_PATH, "*.csv"))
    weights = torch.Tensor()
    df = pd.DataFrame()
    rollout_count = {"opt": 0, "subopt": 0}
    for file in files:
        rollout = pd.read_csv(file, header=0, index_col=None)
        rollout_type = rollout['policy_type'][0]

        if rollout_type == cfg.OPT_LABEL:
            rollout_count["opt"] += len(rollout)
            weights = torch.cat((weights, ratio[0] * torch.ones(len(rollout))))
        else:
            rollout_count["subopt"] += len(rollout)
            weights = torch.cat((weights, ratio[1] * torch.ones(len(rollout))))

        df = pd.concat((df, rollout))

    # Balancing based on opt vs subopt data availability
    if normalize:
        weights[weights == ratio[0]] /= (rollout_count["opt"])
        weights[weights == ratio[1]] /= (rollout_count["subopt"])

    # NOTE: sampled with replacement by default
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    return df, sampler, weights


if __name__ == '__main__':
    # Example: Load rollouts from disk
    data_dir = os.path.join('LunarLanderData')
    filename = 'LunarLanderContinuous-v2_none_0.csv'
    seed = 0
    rollout_name = os.path.join(data_dir, filename)

    # Rollout tuple:
    # <s0,..,s7,a0,a1,r,s'1,...,s'7,done,episode>
    rollout = pd.read_csv(rollout_name, header=0).to_numpy()  # shape: (N x 22)

    # Fixing seeds for reproducible behaviour
    env = gym.make('LunarLanderContinuous-v2')
    env.seed(seed)
    env.action_space.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    for e in range(10):
        score = rollout_viz(env, rollout[rollout[:, 20]==e], seed=int(seed), render=True, overfit=True)
        print("EPISODE {}: SCORE: {}".format(e, score))

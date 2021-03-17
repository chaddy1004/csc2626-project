"""
Experience replay buffer for learning from expert demonstrations
Memory is based on a SumTree ordered according to the TD loss between Q functions
See the following for original implementation:
https://github.com/go2sea/DQfD/blob/951d2c2f5db312bec1390a624d7f6bb7d00b7806/Memory.py 
"""
import os
import sys

import numpy as np
import torch

from config import Config as cfg
from data.data_utils import get_weighted_sampler

sys.path.append(os.path.abspath(os.path.join('..')))
sys.path.append(os.path.abspath(os.path.join('..', 'data')))
# from data.data_utils import *

# np.random.seed(1)
# torch.manual_seed(1)


class SumTree(object):
    data_pointer = 0

    def __init__(self, capacity, permanent_data=0):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)          # ordering is based on TD error
        self.data = np.zeros(capacity, dtype=object)    # data items are <s,a,r,s',done> transitions
        self.permanent_data = permanent_data            # expert data to be kept in the buffer permanently
        assert 0 <= self.permanent_data <= self.capacity
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.data_pointer

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, p)
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.full = True
            # Ensure demo data remains in buffer
            self.data_pointer = self.data_pointer % self.capacity + self.permanent_data

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            if v <= self.tree[left_child_idx]:
                parent_idx = left_child_idx
            else:
                v -= self.tree[left_child_idx]
                parent_idx = right_child_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]


class Memory(object):

    epsilon = 0.001  # small amount to avoid zero priority
    demo_epsilon = 1.0  # 1.0  # extra
    alpha = 0.4  # [0~1] convert the importance of TD error to priority
    beta = 0.6  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity, permanent_data=0, weights=None, offline=False):
        self.permanent_data = permanent_data
        self.tree = SumTree(capacity, permanent_data)
        self.weights = weights
        self.offline = offline

    def __len__(self):
        return len(self.tree)

    def full(self):
        return self.tree.full

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)  # set the max_p for new transition

    def sample(self, n):
        if self.offline:
            assert self.weights is not None
            indices = torch.multinomial(self.weights, n, replacement=True)
            b_memory = self.tree.data[indices]
            # For completeness: compute matching tree indices and set IS to ones
            b_idx = indices + self.tree.capacity - 1 # tree indices
            ISWeights = np.ones((n, 1)) # importance sampling weights
        else:
            assert self.full()
            b_idx = np.empty((n,), dtype=np.int32)
            b_memory = np.empty((n, self.tree.data[0].size), dtype=object)
            ISWeights = np.empty((n, 1)) # importance sampling
            pri_seg = self.tree.total_p / n
            print(self.tree.total_p, pri_seg, n)
            self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

            min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p
            assert min_prob > 0

            for i in range(n):
                v = np.random.uniform(pri_seg * i, pri_seg * (i + 1))
                idx, p, data = self.tree.get_leaf(v)  # note: idx is the index in self.tree.tree
                prob = p / self.tree.total_p
                ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
                b_idx[i], b_memory[i] = idx, data
        return b_idx, b_memory, ISWeights  # note: b_idx stores indexes in self.tree.tree, not in self.tree.data !!!

    # update priority
    def batch_update(self, tree_idxes, abs_errors):
        abs_errors[self.tree.permanent_data:] += self.epsilon
        # priorities of demo transitions are given a bonus of demo_epsilon, to boost the frequency that they are sampled
        abs_errors[:self.tree.permanent_data] += self.demo_epsilon
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idxes, ps):
            self.tree.update(ti, p)


if __name__ == "__main__":
    # Test
    ratio = (0.4, 0.6)
    rollouts_df, weighted_sampler, weights = get_weighted_sampler(ratio, normalize=True)
    rollouts = rollouts_df.to_numpy()
    
    # NOTE: For now, set capacity to length of demo data since should only sample once full
    # although will probably need to reduce size of demo data and increase capacity in the future
    capacity = len(rollouts)

    # If offline: provide weights based on ratio of optimal to suboptimal data
    replay_memory = Memory(capacity=capacity, permanent_data=len(rollouts), weights=weights, offline=True)

    # Adding demo data tuples to memory
    for t in rollouts:
        replay_memory.store(t)

    # Sampling replay memory
    # batch size --> n: number of <s,a,r,s',done,type> sampled from the tree
    # tree_indices: needed (needed only online) to update the tree after each training iteration
    # importance_sampling_weights (needed only online): will be all 1s for now because only demo data in buffer
    tree_indices, minibatch, importance_sampling_weights = replay_memory.sample(cfg.BS)
    print("Batch size: ", len(minibatch))
    print("Tuple size: ", len(minibatch[0]))


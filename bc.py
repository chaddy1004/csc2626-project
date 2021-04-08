import argparse
import os
import random
from collections import namedtuple

import gym
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from ExpertPolicy.network import Actor
from data.data_utils import get_weighted_sampler
from memory.replay_buffer import Memory

sns.set_theme(style="darkgrid")
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

mse_loss_function = torch.nn.MSELoss()

torch.autograd.set_detect_anomaly(True)

if torch.cuda.device_count() > 0:
    print("RUNNING ON GPU")
    DEVICE = torch.device('cuda')
else:
    print("RUNNING ON CPU")
    DEVICE = torch.device('cpu')


class BC:
    def __init__(self, n_states, n_actions, ratio):
        # Test
        self.ratio = ratio
        rollouts_df, weighted_sampler, weights = get_weighted_sampler(self.ratio, normalize=True)
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
        # tree_indices, minibatch, importance_sampling_weights = replay_memory.sample(cfg.BS)
        # hyper parameters

        self.replay_size = 1000000
        self.experience_replay = replay_memory
        self.n_actions = n_actions
        self.n_states = n_states
        self.lr = 0.0003
        self.batch_size = 128
        self.gamma = 0.99
        self.H = -2
        self.reward_scale = 1
        self.policy_eval_start = 0

        # actor network
        self.actor = Actor(n_states=n_states, n_actions=n_actions).to(DEVICE)

        # temperature variable
        self.log_alpha = torch.tensor(0.0, device=DEVICE, requires_grad=True)
        self.optim_alpha = Adam(params=[self.log_alpha], lr=self.lr)
        self.alpha = 0.2

        self.optim_actor = Adam(params=self.actor.parameters(), lr=self.lr)

        self.test_scores = []

    def train_actor(self, s_currs, a_currs, log_action_probs):
        policy_log_prob = self.actor.log_prob(s_currs, a_currs)
        loss_actor = torch.mean(self.alpha * log_action_probs - policy_log_prob)
        self.optim_actor.zero_grad()
        loss_actor.backward()
        self.optim_actor.step()
        return loss_actor

    def train_alpha(self, log_action_probs):
        alpha_loss = torch.mean((-1 * self.log_alpha) * (log_action_probs + self.H).detach())
        self.optim_alpha.zero_grad()
        alpha_loss.backward()
        self.optim_alpha.step()
        self.alpha = torch.exp(self.log_alpha)
        return alpha_loss

    def process_batch(self, x_batch):
        s_currs = x_batch.s_curr
        a_currs = x_batch.a_curr
        r = x_batch.reward
        s_nexts = x_batch.s_next
        dones = x_batch.done
        dones = dones.float()
        return s_currs.to(DEVICE), a_currs.to(DEVICE), r.to(DEVICE), s_nexts.to(DEVICE), dones.to(DEVICE)

    def train(self, x_batch):
        s_currs, a_currs, r, s_nexts, dones = self.process_batch(x_batch=x_batch)
        sample_action, log_action_probs = self.actor.get_action(state=s_currs, train=True)
        alpha_loss = self.train_alpha(log_action_probs=log_action_probs)
        loss_actor = self.train_actor(s_currs=s_currs, a_currs=a_currs, log_action_probs=log_action_probs)

        return loss_actor, alpha_loss


def main(episodes, exp_name, overfit):
    logdir = os.path.join("logs", exp_name)
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)
    env = gym.make('LunarLanderContinuous-v2')
    n_states = env.observation_space.shape[0]  # shape returns a tuple
    n_actions = env.action_space.shape[0]
    agent = BC(n_states=n_states, n_actions=n_actions, ratio=(1.0, 0.0))
    for ep in range(episodes):
        _, data, _ = agent.experience_replay.sample(agent.batch_size)

        sample = namedtuple('sample', ['s_curr', 'a_curr', 'reward', 's_next', 'done'])

        sample.s_curr = torch.from_numpy(data[..., :8])
        sample.a_curr = torch.from_numpy(data[..., 8:10])
        sample.reward = torch.from_numpy(data[..., [10]])
        sample.s_next = torch.from_numpy(data[..., 11:19])
        sample.done = torch.from_numpy(data[..., [19]])

        losses = agent.train(sample)

        if overfit:
            print("OVERFITTING: MAKING SAME ENVIRONMENT")
            env.seed(0)
            env.action_space.seed(0)

        s_curr = env.reset()
        s_curr = np.reshape(s_curr, (1, n_states))
        s_curr = s_curr.astype(np.float32)
        done = False
        score = 0
        step = 0
        # run an episode to see how well it does
        if ep % 100 == 0:
            while not done:
                s_curr_tensor = torch.from_numpy(s_curr)
                a_curr_tensor, _ = agent.actor.get_action(s_curr_tensor.to(DEVICE), train=True)
                # this detach is necessary as the action tensor gets concatenated with state tensor when passed in to critic
                # without this detach, each action tensor keeps its graph, and when same action gets sampled from buffer,
                # it considers that graph "already processed" so it will throw an error
                a_curr_tensor = a_curr_tensor.detach()
                a_curr = a_curr_tensor.cpu().numpy().flatten()

                s_next, r, done, _ = env.step(a_curr)
                s_next = np.reshape(s_next, (1, n_states))
                s_next_tensor = torch.from_numpy(s_next)
                sample = namedtuple('sample', ['s_curr', 'a_curr', 'reward', 's_next', 'done'])
                if step == 500:
                    print("RAN FOR TOO LONG")
                    done = True

                sample.s_curr = s_curr_tensor
                sample.a_curr = a_curr_tensor
                sample.reward = r
                sample.s_next = s_next_tensor
                sample.done = done

                s_curr = s_next
                score += r
                step += 1
                if done:
                    print("ep:{}:################Goal Reached################### {}".format(ep, score))
            env.close()
            writer.add_scalar("score", score, ep)
            writer.add_scalars('training loss', {'loss_actor': losses[0], 'alpha_loss': losses[1]}, ep)
    writer.close()
    return agent


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_name", type=str, default="bc", help="exp_name")
    ap.add_argument("--episodes", type=int, default=10000, help="number of episodes to run")
    ap.add_argument("--overfit", action="store_true", help="number of episodes to run")
    args = vars(ap.parse_args())
    trained_agent = main(episodes=args["episodes"], exp_name=args["exp_name"], overfit=args["overfit"])
    #
    # if DEVICE == torch.device('cpu'):
    #     torch.save(trained_agent.actor, "policy_trained_offline_10_00_on_cpu.pt")
    # else:
    #     torch.save(trained_agent.actor, "policy_trained_offline_10_00_on_gpu.pt")

import argparse
import os
import random
from collections import namedtuple, deque

import gym
import numpy as np
import tensorflow as tf
import torch
from torch.optim import Adam

from ExpertPolicy.network import Actor, Critic
from config import Config as cfg
from data.data_utils import get_weighted_sampler
from memory.replay_buffer import Memory

torch.manual_seed(19971124)
np.random.seed(42)
random.seed(101)

mse_loss_function = torch.nn.MSELoss()

torch.autograd.set_detect_anomaly(True)

if torch.cuda.device_count() > 0:
    print("RUNNING ON GPU")
    DEVICE = torch.device('cuda')
else:
    print("RUNNING ON CPU")
    DEVICE = torch.device('cpu')


class SACOffline:
    def __init__(self, n_states, n_actions):
        # Test
        ratio = (1.0, 0.0)
        rollouts_df, weighted_sampler, weights = get_weighted_sampler(ratio, normalize=True)
        rollouts = rollouts_df.to_numpy()

        # NOTE: For now, set capacity to length of demo data since should only sample once full
        # although will probably need to reduce size of demo data and increase capacity in the future
        capacity = len(rollouts)

        # If offline: provide weights based on ratio of optimal to suboptimal data
        replay_memory = Memory(capacity=capacity, permanent_data=len(rollouts), weights=weights, offline=True)

        # Adding demo data tuples to memory
        for t in rollouts:
            # print(t)
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
        self.Tau = 0.01

        # actor network
        self.actor = Actor(n_states=n_states, n_actions=n_actions).to(DEVICE)

        # dual critic network, with corresponding targets
        self.critic = Critic(n_states=n_states, n_actions=n_actions).to(DEVICE)
        self.critic2 = Critic(n_states=n_states, n_actions=n_actions).to(DEVICE)
        self.target_critic = Critic(n_states=n_states, n_actions=n_actions).to(DEVICE)
        self.target_critic2 = Critic(n_states=n_states, n_actions=n_actions).to(DEVICE)

        # make the target critics start off same as the main networks
        for target_param, local_param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(local_param)

        for target_param, local_param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(local_param)

        # temperature variable
        self.log_alpha = torch.tensor(0.0, device=DEVICE, requires_grad=True)
        self.optim_alpha = Adam(params=[self.log_alpha], lr=self.lr)
        self.alpha = 0.2

        self.optim_actor = Adam(params=self.actor.parameters(), lr=self.lr)
        self.optim_critic = Adam(params=self.critic.parameters(), lr=self.lr)
        self.optim_critic_2 = Adam(params=self.critic2.parameters(), lr=self.lr)

    def get_v(self, state_batch, action, log_action_probs):
        # TODO: move code from main train() function
        return

    def train_actor(self, s_currs):
        # TODO: move code from main train() function
        return

    def train_alpha(self, s_currs, log_action_probs):
        # TODO: move code from main train() function
        return

    def train_critic(self, s_currs, a_currs, r, s_nexts, dones, a_nexts, log_action_probs_next):
        # TODO: move code from main train() function
        return

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

        a_nexts, log_action_probs_next = self.actor.get_action(s_nexts, train=True)

        predicts = self.critic(s_currs, a_currs)  # (batch, actions)
        predicts2 = self.critic2(s_currs, a_currs)

        q_values = self.target_critic(s_nexts, a_nexts).detach()  # (batch, 1)
        q_values_2 = self.target_critic2(s_nexts, a_nexts).detach()
        value = torch.min(q_values, q_values_2) - self.alpha * log_action_probs_next

        target = r + ((1 - dones) * self.gamma * value.detach())
        loss = mse_loss_function(predicts, target)
        self.optim_critic.zero_grad()
        loss.backward()
        self.optim_critic.step()

        loss2 = mse_loss_function(predicts2, target)
        self.optim_critic_2.zero_grad()
        loss2.backward()
        self.optim_critic_2.step()

        sample_action, log_action_probs = self.actor.get_action(state=s_currs, train=True)
        q_values_new = self.critic(s_currs, sample_action)
        q_values_new_2 = self.critic2(s_currs, sample_action)
        loss_actor = (self.alpha * log_action_probs) - torch.min(q_values_new, q_values_new_2)

        loss_actor = torch.mean(loss_actor)
        self.optim_actor.zero_grad()
        loss_actor.backward()
        self.optim_actor.step()

        alpha_loss = torch.mean((-1 * torch.exp(self.log_alpha)) * (log_action_probs.detach() + self.H))
        self.optim_alpha.zero_grad()
        alpha_loss.backward()
        self.optim_alpha.step()
        self.alpha = torch.exp(self.log_alpha)
        self.update_weights()
        return

    def update_weights(self):
        for target_param, local_param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.Tau * local_param.data + (1.0 - self.Tau) * target_param.data)

        for target_param, local_param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.Tau * local_param.data + (1.0 - self.Tau) * target_param.data)


def main(episodes, exp_name, offline):
    logdir = os.path.join("logs", exp_name)
    os.makedirs(logdir, exist_ok=True)
    writer = tf.summary.create_file_writer(logdir)
    env = gym.make('LunarLanderContinuous-v2')
    n_states = env.observation_space.shape[0]  # shape returns a tuple
    n_actions = env.action_space.shape[0]
    agent = SACOffline(n_states=n_states, n_actions=n_actions)
    for ep in range(episodes):
        _, data, _ = agent.experience_replay.sample(agent.batch_size)

        sample = namedtuple('sample', ['s_curr', 'a_curr', 'reward', 's_next', 'done'])

        s_curr_tensor = torch.from_numpy(data[..., :8])
        a_curr_tensor = torch.from_numpy(data[..., 8:10])
        r = torch.from_numpy(data[..., [10]])
        s_next_tensor = torch.from_numpy(data[..., 11:19])
        done = torch.from_numpy(data[..., [19]])

        sample.s_curr = s_curr_tensor
        sample.a_curr = a_curr_tensor
        sample.reward = r
        sample.s_next = s_next_tensor
        sample.done = done

        agent.train(sample)

        s_curr = env.reset()
        s_curr = np.reshape(s_curr, (1, n_states))
        s_curr = s_curr.astype(np.float32)
        done = False
        score = 0
        step = 0

        # run an episode to see how well it does
        while not done:
            s_curr_tensor = torch.from_numpy(s_curr)
            a_curr_tensor, _ = agent.actor.get_action(s_curr_tensor.to(DEVICE), train=True)
            # this detach is necessary as the action tensor gets concatenated with state tensor when passed in to critic
            # without this detach, each action tensor keeps its graph, and when same action gets sampled from buffer,
            # it considers that graph "already processed" so it will throw an error
            a_curr_tensor = a_curr_tensor.detach()
            a_curr = a_curr_tensor.cpu().numpy().flatten()

            if offline:
                s_next, r, done, _ = env.step(a_curr)
            else:
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
                print(f"ep:{ep}:################Goal Reached###################", score)
                with writer.as_default():
                    tf.summary.scalar("reward", r, ep)
                    tf.summary.scalar("score", score, ep)
    return agent


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_name", type=str, default="sac_offline", help="exp_name")
    ap.add_argument("--episodes", type=int, default=700, help="number of episodes to run")
    ap.add_argument("--offline", action="store_true", help="number of episodes to run")
    args = vars(ap.parse_args())
    trained_agent = main(episodes=args["episodes"], exp_name=args["exp_name"], offline=args["exp_name"])
    if DEVICE == torch.device('cpu'):
        torch.save(trained_agent.actor, "policy_trained_on_cpu.pt")
    else:
        torch.save(trained_agent.actor, "policy_trained_on_gpu.pt")

import argparse
import os
import random
from collections import namedtuple, deque

import gym
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from ExpertPolicy.network import Actor, Critic
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


class DDPGOffline:
    def __init__(self, n_states, n_actions):
        # Test
        # hyper parameters
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

        self.replay_size = 1000000
        self.experience_replay = replay_memory
        self.n_actions = n_actions
        self.n_states = n_states
        self.lr = 0.0003
        self.batch_size = 128
        self.gamma = 0.99
        self.Tau = 0.01

        self.n_random_actions = 10
        # actor network
        self.actor = Actor(n_states=n_states, n_actions=n_actions).to(DEVICE)

        # dual critic network, with corresponding targets
        self.critic = Critic(n_states=n_states, n_actions=n_actions).to(DEVICE)
        self.target_critic = Critic(n_states=n_states, n_actions=n_actions).to(DEVICE)

        # make the target critics start off same as the main networks
        for target_param, local_param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(local_param)

        # temperature variable
        self.log_alpha = torch.tensor(0.0, device=DEVICE, requires_grad=True)
        self.optim_alpha = Adam(params=[self.log_alpha], lr=self.lr)
        self.alpha = 0.2

        self.optim_actor = Adam(params=self.actor.parameters(), lr=self.lr)
        self.optim_critic = Adam(params=self.critic.parameters(), lr=self.lr)

    def train_actor(self, s_currs):
        action, _ = self.actor.get_action(s_currs, train=True)
        q_values_new = self.critic(s_currs, action)
        loss_actor = -torch.mean(q_values_new)
        self.optim_actor.zero_grad()
        loss_actor.backward()
        self.optim_actor.step()
        return

    def train_critic(self, s_currs, a_currs, r, s_nexts, dones):
        predicts = self.critic(s_currs, a_currs)  # (batch, actions)
        a_next, _ = self.actor.get_action(s_nexts, train=True)
        predict_targ = self.target_critic(s_nexts, a_next.detach())
        target = r + ((1. - dones) * self.gamma * predict_targ)

        loss = mse_loss_function(predicts, target.detach())

        self.optim_critic.zero_grad()
        loss.backward()
        self.optim_critic.step()
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
        self.train_critic(s_currs=s_currs, a_currs=a_currs, r=r, s_nexts=s_nexts, dones=dones)
        self.train_actor(s_currs=s_currs)
        self.update_weights()
        return

    def update_weights(self):
        for target_param, local_param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.Tau * local_param.data + (1.0 - self.Tau) * target_param.data)


class DDPG:
    def __init__(self, n_states, n_actions):
        # Test
        # hyper parameters
        self.replay_size = 1000000
        self.experience_replay = deque(maxlen=self.replay_size)
        self.n_actions = n_actions
        self.n_states = n_states
        self.lr = 0.0003
        self.batch_size = 128
        self.gamma = 0.99
        self.Tau = 0.01

        self.n_random_actions = 10
        # actor network
        self.actor = Actor(n_states=n_states, n_actions=n_actions).to(DEVICE)

        # dual critic network, with corresponding targets
        self.critic = Critic(n_states=n_states, n_actions=n_actions).to(DEVICE)
        self.target_critic = Critic(n_states=n_states, n_actions=n_actions).to(DEVICE)

        # make the target critics start off same as the main networks
        for target_param, local_param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(local_param)

        # temperature variable
        self.log_alpha = torch.tensor(0.0, device=DEVICE, requires_grad=True)
        self.optim_alpha = Adam(params=[self.log_alpha], lr=self.lr)
        self.alpha = 0.2

        self.optim_actor = Adam(params=self.actor.parameters(), lr=self.lr)
        self.optim_critic = Adam(params=self.critic.parameters(), lr=self.lr)

    def train_actor(self, s_currs):
        action, _ = self.actor.get_action(s_currs, train=True)
        q_values_new = self.critic(s_currs, action)
        loss_actor = -torch.mean(q_values_new)
        self.optim_actor.zero_grad()
        loss_actor.backward()
        self.optim_actor.step()
        return

    def train_critic(self, s_currs, a_currs, r, s_nexts, dones):
        predicts = self.critic(s_currs, a_currs)  # (batch, actions)
        a_next, _ = self.actor.get_action(s_nexts, train=True)
        predict_targ = self.target_critic(s_nexts, a_next.detach())
        target = r + ((1. - dones) * self.gamma * predict_targ)

        loss = mse_loss_function(predicts, target.detach())

        self.optim_critic.zero_grad()
        loss.backward()
        self.optim_critic.step()
        return

    def process_batch(self, x_batch):
        s_currs = torch.zeros((self.batch_size, self.n_states))
        a_currs = torch.zeros((self.batch_size, self.n_actions))
        r = torch.zeros((self.batch_size, 1))
        s_nexts = torch.zeros((self.batch_size, self.n_states))
        dones = torch.zeros((self.batch_size, 1))

        for batch in range(self.batch_size):
            s_currs[batch] = x_batch[batch].s_curr
            a_currs[batch] = x_batch[batch].a_curr
            r[batch] = x_batch[batch].reward
            s_nexts[batch] = x_batch[batch].s_next
            dones[batch] = x_batch[batch].done
        dones = dones.float()
        return s_currs.to(DEVICE), a_currs.to(DEVICE), r.to(DEVICE), s_nexts.to(DEVICE), dones.to(DEVICE)

    def train(self, x_batch):
        s_currs, a_currs, r, s_nexts, dones = self.process_batch(x_batch=x_batch)
        self.train_critic(s_currs=s_currs, a_currs=a_currs, r=r, s_nexts=s_nexts, dones=dones)
        self.train_actor(s_currs=s_currs)
        self.update_weights()
        return

    def update_weights(self):
        for target_param, local_param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.Tau * local_param.data + (1.0 - self.Tau) * target_param.data)


def main(episodes, exp_name, offline):
    logdir = os.path.join("logs", exp_name)
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)
    env = gym.make('LunarLanderContinuous-v2')
    n_states = env.observation_space.shape[0]  # shape returns a tuple
    n_actions = env.action_space.shape[0]
    agent = DDPG(n_states=n_states, n_actions=n_actions)
    exploration_eps = -1
    for ep in range(episodes):
        s_curr = env.reset()
        s_curr = np.reshape(s_curr, (1, n_states))
        s_curr = s_curr.astype(np.float32)
        done = False
        score = 0
        step = 0
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
            # must re-make training dataloader since the dataset is now updated with aggregation of new data
            sample.s_curr = s_curr_tensor
            sample.a_curr = a_curr_tensor
            sample.reward = r
            sample.s_next = s_next_tensor
            sample.done = done

            if len(agent.experience_replay) < agent.batch_size:
                agent.experience_replay.append(sample)
                print("appending to buffer....")
            else:
                agent.experience_replay.append(sample)
                if ep > exploration_eps:
                    x_batch = random.sample(agent.experience_replay, agent.batch_size)
                    agent.train(x_batch)

            s_curr = s_next
            score += r
            step += 1
            if done:
                print(f"ep:{ep}:################Goal Reached###################", score)
                if ep > 0:
                    writer.add_scalar("score", score, ep)

    writer.close()
    return agent


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_name", type=str, default="DDPG_LunarLander_Score", help="exp_name")
    ap.add_argument("--episodes", type=int, default=700, help="number of episodes to run")
    args = vars(ap.parse_args())
    trained_agent = main(episodes=args["episodes"], exp_name=args["exp_name"], offline=args["exp_name"])
    if DEVICE == torch.device('cpu'):
        torch.save(trained_agent.actor, "policy_trained_on_cpu.pt")
    else:
        torch.save(trained_agent.actor, "policy_trained_on_gpu.pt")

import argparse
import os
import random
from collections import namedtuple

import gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.optim import Adam

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


class CQLDDPG:
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
            # print(t)
            replay_memory.store(t)

        self.replay_size = 1000000
        self.experience_replay = replay_memory
        self.n_actions = n_actions
        self.n_states = n_states
        self.lr = 0.0003
        self.batch_size = 128
        self.gamma = 0.99
        self.H = -2
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

        self.test_scores = []

    def train_actor(self, s_currs):
        action, _ = self.actor(s_currs)
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

        loss_rl = mse_loss_function(predicts, target.detach())

        random_actions = torch.FloatTensor(predicts.shape[0] * self.n_random_actions, a_currs.shape[-1]).uniform_(-1, 1)
        random_actions = random_actions.to(DEVICE)

        s_currs_temp = s_currs.unsqueeze(1).repeat(1, self.n_random_actions, 1).view(
            s_currs.shape[0] * self.n_random_actions, s_currs.shape[1])
        curr_action_tensor, curr_log_action_probs = self.actor.get_action(state=s_currs_temp, train=True)
        s_nexts_temp = s_nexts.unsqueeze(1).repeat(1, self.n_random_actions, 1).view(
            s_nexts.shape[0] * self.n_random_actions, s_nexts.shape[1])
        new_curr_action_tensor, new_log_action_probs = self.actor.get_action(state=s_nexts_temp, train=True)

        rand = self.critic.get_tensor_values(s_currs, random_actions)

        curr_actions = self.critic.get_tensor_values(s_currs, curr_action_tensor.detach())

        next_actions = self.critic.get_tensor_values(s_currs, new_curr_action_tensor.detach())

        # min_q (CQL term #1)
        # NOTE: assuming min_q_weight (i.e. alpha) is 1
        random_density_log = np.log(0.5 ** self.n_actions)

        rand_value_1 = rand - random_density_log

        pred_value_1 = curr_actions - curr_log_action_probs.view(s_currs.shape[0], self.n_random_actions, 1).detach()

        pred_value_1_next = next_actions - new_log_action_probs.view(s_nexts.shape[0], self.n_random_actions,
                                                                     1).detach()

        q1_term_pre_exp = torch.cat([rand_value_1, pred_value_1, pred_value_1_next], 1)

        loss_cql_1 = torch.logsumexp(q1_term_pre_exp, dim=1).mean()

        loss_cql_1 = loss_cql_1 - predicts.mean()

        # Bellman + CQL
        loss = loss_rl + loss_cql_1

        self.optim_critic.zero_grad()
        RETAIN_G = False
        loss.backward(retain_graph=RETAIN_G)
        self.optim_critic.step()

        return loss

    def process_batch(self, x_batch):
        s_currs = x_batch.s_curr
        a_currs = x_batch.a_curr
        r = x_batch.reward
        s_nexts = x_batch.s_next
        dones = x_batch.done
        dones = dones.float()
        return s_currs.to(DEVICE), a_currs.to(DEVICE), r.to(DEVICE), s_nexts.to(DEVICE), dones.to(DEVICE)

    def train(self, x_batch, ep):
        s_currs, a_currs, r, s_nexts, dones = self.process_batch(x_batch=x_batch)
        self.train_critic(s_currs=s_currs, a_currs=a_currs, r=r, s_nexts=s_nexts, dones=dones)
        self.train_actor(s_currs=s_currs)
        self.update_weights()
        return

    def update_weights(self):
        for target_param, local_param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.Tau * local_param.data + (1.0 - self.Tau) * target_param.data)


class CQLSAC:
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

        self.n_random_actions = 10
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

        self.test_scores = []

    def get_v(self, state_batch):
        action_batch, log_action_probs = self.actor.get_action(state_batch, train=True)
        q_values = self.target_critic(state_batch, action_batch).detach()  # (batch, 1)
        q_values_2 = self.target_critic2(state_batch, action_batch).detach()
        value = torch.min(q_values, q_values_2) - self.alpha * log_action_probs
        return value.detach()

    def train_actor(self, s_currs, sample_action, log_action_probs):
        q_values_new = self.critic(s_currs, sample_action)
        q_values_new_2 = self.critic2(s_currs, sample_action)
        loss_actor = (self.alpha * log_action_probs) - torch.min(q_values_new, q_values_new_2)

        loss_actor = torch.mean(loss_actor)
        self.optim_actor.zero_grad()
        loss_actor.backward()
        self.optim_actor.step()
        return

    def train_alpha(self, log_action_probs):
        alpha_loss = torch.mean((-1 * torch.exp(self.log_alpha)) * (log_action_probs.detach() + self.H))
        self.optim_alpha.zero_grad()
        alpha_loss.backward()
        self.optim_alpha.step()
        with torch.no_grad():
            self.alpha = torch.exp(self.log_alpha)
        return

    def _get_tensor_values(self, obs, actions, network=None):
        action_shape = actions.shape[0]
        obs_shape = obs.shape[0]
        num_repeat = int(action_shape / obs_shape)
        obs_temp = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(
            obs.shape[0] * num_repeat, obs.shape[1])
        preds = network(obs_temp, actions)
        preds = preds.view(obs.shape[0], num_repeat, 1)
        return preds

    def train_critic(self, value, s_currs, a_currs, r, s_nexts, dones):
        predicts = self.critic(s_currs, a_currs)  # (batch, actions)
        predicts2 = self.critic2(s_currs, a_currs)
        target = r + ((1. - dones) * self.gamma * value)

        loss_rl = mse_loss_function(predicts, target.detach())
        loss2_rl = mse_loss_function(predicts2, target.detach())

        random_actions = torch.FloatTensor(predicts.shape[0] * self.n_random_actions, a_currs.shape[-1]).uniform_(-1, 1)
        random_actions = random_actions.to(DEVICE)

        s_currs_temp = s_currs.unsqueeze(1).repeat(1, self.n_random_actions, 1).view(
            s_currs.shape[0] * self.n_random_actions, s_currs.shape[1])
        curr_action_tensor, curr_log_action_probs = self.actor.get_action(state=s_currs_temp, train=True)
        s_nexts_temp = s_nexts.unsqueeze(1).repeat(1, self.n_random_actions, 1).view(
            s_nexts.shape[0] * self.n_random_actions, s_nexts.shape[1])
        new_curr_action_tensor, new_log_action_probs = self.actor.get_action(state=s_nexts_temp, train=True)

        rand = self.critic.get_tensor_values(s_currs, random_actions)
        rand2 = self.critic2.get_tensor_values(s_currs, random_actions)

        curr_actions = self.critic.get_tensor_values(s_currs, curr_action_tensor.detach())
        curr_actions2 = self.critic2.get_tensor_values(s_currs, curr_action_tensor.detach())

        next_actions = self.critic.get_tensor_values(s_currs, new_curr_action_tensor.detach())
        next_actions2 = self.critic2.get_tensor_values(s_currs, new_curr_action_tensor.detach())

        # min_q (CQL term #1)
        # NOTE: assuming min_q_weight (i.e. alpha) is 1
        random_density_log = np.log(0.5 ** self.n_actions)

        rand_value_1 = rand - random_density_log
        rand_value_2 = rand2 - random_density_log

        pred_value_1 = curr_actions - curr_log_action_probs.view(s_currs.shape[0], self.n_random_actions, 1).detach()
        pred_value_2 = curr_actions2 - curr_log_action_probs.view(s_currs.shape[0], self.n_random_actions, 1).detach()

        pred_value_1_next = next_actions - new_log_action_probs.view(s_nexts.shape[0], self.n_random_actions,
                                                                     1).detach()
        pred_value_2_next = next_actions2 - new_log_action_probs.view(s_nexts.shape[0], self.n_random_actions,
                                                                      1).detach()

        q1_term_pre_exp = torch.cat([rand_value_1, pred_value_1, pred_value_1_next], 1)
        q2_term_pre_exp = torch.cat([rand_value_2, pred_value_2, pred_value_2_next], 1)

        loss_cql_1 = torch.logsumexp(q1_term_pre_exp, dim=1).mean()
        loss_cql_2 = torch.logsumexp(q2_term_pre_exp, dim=1).mean()

        loss_cql_1 = loss_cql_1 - predicts.mean()
        loss_cql_2 = loss_cql_2 - predicts2.mean()

        # Bellman + CQL
        loss = loss_rl + loss_cql_1
        loss2 = loss2_rl + loss_cql_2

        self.optim_critic.zero_grad()
        RETAIN_G = False
        loss.backward(retain_graph=RETAIN_G)
        self.optim_critic.step()
        self.optim_critic_2.zero_grad()
        loss2.backward(retain_graph=RETAIN_G)
        self.optim_critic_2.step()
        return loss, loss2

    def process_batch(self, x_batch):
        s_currs = x_batch.s_curr
        a_currs = x_batch.a_curr
        r = x_batch.reward
        s_nexts = x_batch.s_next
        dones = x_batch.done
        dones = dones.float()
        return s_currs.to(DEVICE), a_currs.to(DEVICE), r.to(DEVICE), s_nexts.to(DEVICE), dones.to(DEVICE)

    def train(self, x_batch, ep):
        s_currs, a_currs, r, s_nexts, dones = self.process_batch(x_batch=x_batch)
        sample_action, log_action_probs = self.actor.get_action(state=s_currs, train=True)
        self.train_alpha(log_action_probs=log_action_probs)
        self.train_actor(s_currs=s_currs, sample_action=sample_action, log_action_probs=log_action_probs)
        value = self.get_v(state_batch=s_nexts)
        self.train_critic(value=value, s_currs=s_currs, a_currs=a_currs, r=r, s_nexts=s_nexts, dones=dones)

        self.update_weights()
        return

    def update_weights(self):
        for target_param, local_param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.Tau * local_param.data + (1.0 - self.Tau) * target_param.data)

        for target_param, local_param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.Tau * local_param.data + (1.0 - self.Tau) * target_param.data)


def main(episodes, exp_name, offline, overfit):
    logdir = os.path.join("logs", exp_name)
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)
    env = gym.make('LunarLanderContinuous-v2')
    n_states = env.observation_space.shape[0]  # shape returns a tuple
    n_actions = env.action_space.shape[0]
    agent = CQLSAC(n_states=n_states, n_actions=n_actions, ratio=(1.0, 0.0))
    # agent = CQLDDPG(n_states=n_states, n_actions=n_actions, ratio=(1.0, 0.0))
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
            # env.render()
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
                writer.add_scalar("reward", r, ep)
                writer.add_scalar("score", score, ep)
    return agent


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_name", type=str, default="saccql_with_is", help="exp_name")
    ap.add_argument("--episodes", type=int, default=10000, help="number of episodes to run")
    ap.add_argument("--offline", action="store_true", help="number of episodes to run")
    ap.add_argument("--overfit", action="store_true", help="number of episodes to run")
    args = vars(ap.parse_args())
    trained_agent = main(episodes=args["episodes"], exp_name=args["exp_name"], offline=args["exp_name"],
                         overfit=args["overfit"])
    if DEVICE == torch.device('cpu'):
        torch.save(trained_agent.actor, "policy_trained_offline_10_00_on_cpu.pt")
    else:
        torch.save(trained_agent.actor, "policy_trained_offline_10_00_on_gpu.pt")

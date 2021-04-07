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

from sac import SAC
from sac_offline import SACOffline

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


def main(episodes, exp_name, agent_type):
    env = gym.make('LunarLanderContinuous-v2')
    n_states = env.observation_space.shape[0]  # shape returns a tuple
    n_actions = env.action_space.shape[0]
    agent = None
    expert_data_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for expert_data_ratio in expert_data_ratios:
        logdir = os.path.join("logs", f"{exp_name}_{expert_data_ratio}")
        os.makedirs(logdir, exist_ok=True)
        writer = tf.summary.create_file_writer(logdir)

        ratio = (expert_data_ratio, 1.0 - expert_data_ratio)
        if agent_type == "SACOffline":
            agent = SACOffline(n_states=n_states, n_actions=n_actions, ratio=ratio)
        elif agent_type == "BC":
            agent = None
        elif agent_type == "CQL":
            agent = None

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

            # testing on environment
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
                    with writer.as_default():
                        tf.summary.scalar("score", score, ep)
    return agent


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_name", type=str, default="sac_offline_small_data_10_00", help="exp_name")
    ap.add_argument("--episodes", type=int, default=1000, help="number of episodes to run")
    ap.add_argument("--agent_type", type=str, default="BC", help="number of episodes to run")
    args = vars(ap.parse_args())
    trained_agent = main(episodes=args["episodes"], exp_name=args["exp_name"], agent_type=args["agent_type"])
    if DEVICE == torch.device('cpu'):
        torch.save(trained_agent.actor, "policy_trained_offline_10_00_on_cpu.pt")
    else:
        torch.save(trained_agent.actor, "policy_trained_offline_10_00_on_gpu.pt")

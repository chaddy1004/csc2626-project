import argparse
import os
import random
from collections import namedtuple

import gym
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from bc import BC
from sac_offline import SACOffline
from ddpg_offline import DDPGOffline
from cql import CQLSAC, CQLDDPG

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


def generate_plot(episodes, expert_ratios, scores, colour_len, log_freq, agent_type):
    d = {"Episode": episodes, "ExpertRatio": expert_ratios}
    experiments = pd.DataFrame(data=d)
    experiments.insert(1, "Score", scores, allow_duplicates=False)
    plot = sns.lineplot(data=experiments, x="Episode", y="Score", hue="ExpertRatio", legend="full",
                        palette=sns.color_palette("hls", colour_len))
    plot.set_xticks(range(0, max(episodes), log_freq))
    plot.set_title(f"Scores of {agent_type}")

    sns.lineplot(x=[ep for ep in range(0, max(episodes) + log_freq, log_freq)],
                 y=[0 for _ in range(0, max(episodes) + log_freq, log_freq)],
                 color="black", linestyle="--")

    sns.lineplot(x=[ep for ep in range(0, max(episodes) + log_freq, log_freq)],
                 y=[200 for _ in range(0, max(episodes) + log_freq, log_freq)],
                 color="black", linestyle="--")

    _path = os.path.join("experiments", "figures")
    if not os.path.isdir(_path):
        os.makedirs(_path, exist_ok=False)
    figname = f"{agent_type}.png"
    path = os.path.join(_path, figname)
    plot.figure.savefig(path)


def main(episodes, agent_type, num_trials):
    env = gym.make('LunarLanderContinuous-v2')
    n_states = env.observation_space.shape[0]  # shape returns a tuple
    n_actions = env.action_space.shape[0]
    agent = None
    EXPERT_DATA_RATIOS = [0.0, 0.3, 0.5, 0.7, 1.0]
    # EXPERT_DATA_RATIOS = [0.0, 1.0]
    log_freq = 100

    total_episodes = []
    total_expert_ratios = []
    total_scores = []
    for expert_data_ratio in EXPERT_DATA_RATIOS:
        print(F"EXPERT RATIO: {expert_data_ratio}")
        per_ratio_scores = []
        per_ratio_logged_episodes = []
        per_ratio_trials = []
        per_ratio_logged_expert_ratios = []
        for trial in range(num_trials):
            print(F"CURR TRIAL: {trial}")
            per_ratio_trials.append(trial)
            ratio = (expert_data_ratio, 1.0 - expert_data_ratio)
            if agent_type == "SACOffline":
                agent = SACOffline(n_states=n_states, n_actions=n_actions, ratio=ratio)
            elif agent_type == "DDPGOffline":
                agent = DDPGOffline(n_states=n_states, n_actions=n_actions, ratio=ratio)
            elif agent_type == "BC":
                agent = BC(n_states=n_states, n_actions=n_actions, ratio=ratio)
            elif agent_type == "CQLSAC":
                agent = CQLSAC(n_states=n_states, n_actions=n_actions, ratio=ratio)
            elif agent_type == "CQLDDPG":
                agent = CQLDDPG(n_states=n_states, n_actions=n_actions, ratio=ratio)

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

                agent.train(sample, ep)

                # testing on environment
                s_curr = env.reset()
                s_curr = np.reshape(s_curr, (1, n_states))
                s_curr = s_curr.astype(np.float32)
                done = False
                score = 0
                step = 0
                # run an episode to see how well it does
                if ep % log_freq == 0:
                    per_ratio_logged_episodes.append(ep)
                    per_ratio_logged_expert_ratios.append(expert_data_ratio)
                    while not done:
                        s_curr_tensor = torch.from_numpy(s_curr)
                        a_curr_pred, _ = agent.actor.get_action(s_curr_tensor.to(DEVICE), train=False)
                        a_curr = np.squeeze(a_curr_pred)

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
                            agent.test_scores.append(score)
            per_ratio_scores += agent.test_scores

        total_episodes += per_ratio_logged_episodes
        total_expert_ratios += per_ratio_logged_expert_ratios
        total_scores += per_ratio_scores

    generate_plot(episodes=total_episodes, expert_ratios=total_expert_ratios, scores=total_scores,
                  colour_len=len(EXPERT_DATA_RATIOS), log_freq=log_freq, agent_type=agent_type)

    return agent


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300, help="number of episodes to run")
    ap.add_argument("--agent_type", type=str, default="CQLSAC", help="type of agent to test")
    ap.add_argument("--num_trials", type=int, default=2, help="number of trials to average over")
    args = vars(ap.parse_args())
    agent_type = args["agent_type"]
    episodes = args["episodes"]
    num_trials = args["num_trials"]
    trained_agent = main(episodes=episodes, agent_type=agent_type, num_trials=num_trials)

    _path = os.path.join("experiments", "weights")

    if not os.path.isdir(_path):
        os.makedirs(_path, exist_ok=False)

    weight_name = f"{agent_type}_trained_for_{episodes}.pt"
    path = os.path.join(_path, weight_name)
    torch.save(trained_agent.actor, path)

import argparse
from pathlib import Path
from functools import partial

import gym
import numpy as np
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--env', type=str, default='LunarLanderContinuous-v2')
    parser.add_argument('--policy_file', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='output/')
    parser.add_argument('--noise_type', choices=['none', 'uniform', 'gaussian'], default='none')
    parser.add_argument('--num_episodes', type=int, default=1)
    parser.add_argument('--max_iterations', type=int, default=1000)
    args = parser.parse_args()

    assert args.num_episodes > 0

    return args

class Noise:
    def __init__(self):
        pass

    def sample_noise(self):
        raise NotImplementedError


class UniformNoise:
    def __init__(self, cfg):
        super().__init__()

    def sample_noise(self):
        pass


def get_policy(policy_file):
    return None


def get_episode_data(env, policy=None, noise=None, max_iterations=1000):
    env.reset()

    # Record data
    states = []
    actions = []
    rewards = []
    dones = []

    for _ in range(max_iterations):
        env.render()
        # TOOD (get policy here)
        if policy is None:
            action = env.action_space.sample()
        else:
            raise NotImplementedError

        state, reward, done, _ = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        if done:
            break
    env.close()

    states, actions, rewards, dones = map(np.array, (states, actions, rewards, dones))

    curr_states = states[:-1]
    next_states = states[1:]
    actions = actions[:-1]
    rewards = rewards[:-1]
    dones = dones[1:]

    return curr_states, next_states, actions, rewards, dones


if __name__ == '__main__':
    args = get_args()
    print(args)

    env = gym.make(args.env)
    env.seed(args.seed)
    np.random.seed(args.seed)

    df_all = pd.DataFrame()
    policy = get_policy(args.policy_file)

    for episode in range(args.num_episodes):
        curr_states, next_states, actions, rewards, dones = get_episode_data(env, policy=policy, noise=args.noise_type, max_iterations=args.max_iterations)

        df_curr_states = pd.DataFrame(data=curr_states, columns=[f'curr_state_{i}' for i in range(curr_states.shape[-1])])
        df_actions = pd.DataFrame(data=actions, columns=[f'action_{i}' for i in range(actions.shape[-1])])
        df_actions['reward'] = rewards
        df_next_states = pd.DataFrame(data=next_states, columns=[f'next_state_{i}' for i in range(next_states.shape[-1])])
        df_next_states['done'] = dones*1
        df_next_states['episode'] = episode

        df_episode = pd.concat([df_curr_states, df_actions, df_next_states], axis=1)
        df_all = pd.concat([df_all, df_episode])

    print(df_all.head())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'{args.env}_{args.noise_type}_{args.num_episodes}.csv'
    df_all.to_csv(output_file, index=False)

from torch.optim import Adam
from network import Actor, Critic
import torch

import numpy as np

import gym

import argparse

JUMP = 1
UP = 2
RIGHT = 3
LEFT = 4
DOWN = 5

GAME_ACTION_SIZES = {"MontezumaRevenge": 5}


class SAC:
    def __init__(self, state_shape, n_actions):
        self.states = []  # logging the states
        self.states_next = []  # logging the states
        self.actions = []  # logging the ACTUAL action that was performed at t
        self.model_outputs = []  # logging PI(a|s) that was outputted at t
        self.rewards = []  # logging the rewards that go from t_i -> t_{i+1}
        self.n_actions = n_actions
        self.lr = 0.001
        self.batch_size = 64
        self.gamma = 0.99
        self.actor = Actor(state_shape=state_shape, n_actions=n_actions)
        self.critic = Critic(state_shape=state_shape)
        self.optim_actor = Adam(params=self.actor.parameters(), lr=self.lr)
        self.optim_critic = Adam(params=self.critic.parameters(), lr=self.lr * 5)

    def reset(self):
        self.states = []
        self.states_next = []
        self.actions = []
        self.model_outputs = []
        self.rewards = []

    def get_action(self, state):
        action_probs = self.actor(state.float())
        action_probs_np = action_probs.detach().cpu().numpy().squeeze()
        return int(np.random.choice(self.n_actions, 1, p=action_probs_np)), action_probs

    def get_action_probs(self):
        # this is used for REINFORCE and A2C. Not sure if this will be used for SAC
        model_outputs_tensor = torch.cat(self.model_outputs, 0)
        index = torch.Tensor(self.actions).long()
        chosen_actions_tensor = torch.zeros_like(model_outputs_tensor).long()
        chosen_actions_tensor[torch.arange(chosen_actions_tensor.size(0)), index] = 1.
        action_probs_tensor = model_outputs_tensor * chosen_actions_tensor
        # print(model_outputs_tensor)
        # print(action_probs_tensor)
        action_probs_tensor_flattened = torch.sum(action_probs_tensor, dim=1).unsqueeze(1)
        return action_probs_tensor_flattened

    def train_critic(self):
        # TODO
        pass

    def train_actor(self):
        # TODO
        pass


def main_training_loop(env, agent):
    env.reset()
    agent.reset()
    # TODO
    pass


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--game_name", type=str, default="MontezumaRevenge", help="game_name")

    args = vars(ap.parse_args())
    game_name = args["game_name"]
    env_name = f"{game_name}-v0"
    env = gym.make(env_name)
    print(f"Env: {env_name}")

    print(env.observation_space.shape, type(env.observation_space.shape), env.action_space)

    agent = SAC(state_shape=env.observation_space.shape, n_actions=GAME_ACTION_SIZES[game_name])

    main_training_loop(env=env, agent=agent)

    # """
    # 1 = jump
    # 2 = up
    # 3 = right
    # 4 = left
    # 5= down
    # """
    #
    # while True:
    #     env.render()
    #     action = input("pick between 0-17")
    #     env.step(int(action))

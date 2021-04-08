import numpy as np


class Policy:
    def __init__(self, policy):
        self.policy = policy

    def get_action(self, state_tensor, train=False):
        action, log_pi = self.policy.get_action(state_tensor, train=train)
        return action


class CorruptedPolicy(Policy):
    def __init__(self, policy):
        super().__init__(policy)

    def get_action(self, state_tensor, train=False):
        action = super().get_action(state_tensor, train)
        return self.corrupt_action(action)

    def corrupt_action(action):
        raise NotImplementedError


class NoisyPolicy(CorruptedPolicy):
    """
    With a probability of 30%, sample a random action
    """
    def __init__(self, policy, env):
        super().__init__(policy)
        self.env = env
        self.noise_prob = 0.3

    def corrupt_action(self, action):
        is_noisy = np.random.choice([True, False], p=[self.noise_prob, 1. - self.noise_prob])

        if is_noisy:
            action = self.env.action_space.sample()

        return action


class LaggyPolicy(CorruptedPolicy):
    def __init__(self, policy):
        super().__init__(policy)
        self.laggy_prob = 0.85
        self.num_prev_states = 1
        self.prev_states = [None] * self.num_prev_states
        self.index = 0

    def store_action(self, action):
        self.prev_states[self.index] = action
        self.index = (self.index + 1) % self.num_prev_states

    def get_prev_action(self, num_states_back):
        index = (self.index - num_states_back) % self.num_prev_states
        return self.prev_states[index]

    def corrupt_action(self, action):
        is_laggy = np.random.choice([True, False], p=[self.laggy_prob, 1. - self.laggy_prob])

        if is_laggy:
            laggy_action = self.get_prev_action(self.num_prev_states)
            if isinstance(laggy_action, np.ndarray):
                action = laggy_action
        self.store_action(action)

        return action

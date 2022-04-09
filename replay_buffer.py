import numpy as np

class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.m_size = max_size
        self.m_cntr = 0
        self.state_m = np.zeros((self.m_size, *input_shape))
        self.new_state_m = np.zeros((self.m_size, *input_shape))
        self.action_m = np.zeros((self.m_size, n_actions))
        self.reward_m = np.zeros(self.m_size)
        self.terminal_m = np.zeros(self.m_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.m_cntr % self.m_size

        self.state_m[index] = state
        self.new_state_m[index] = state_
        self.action_m[index] = action
        self.reward_m[index] = reward
        self.terminal_m[index] = done

        self.m_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.m_cntr, self.m_size)

        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_m[batch]
        states_ = self.new_state_m[batch]
        actions = self.action_m[batch]
        rewards = self.reward_m[batch]
        dones = self.terminal_m[batch]

        return states, actions, rewards, states_, dones
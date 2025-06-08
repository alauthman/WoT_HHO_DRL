import gym
from gym import spaces
import numpy as np
from .utils import load_dataset

class WoTEnv(gym.Env):
    def __init__(self, dataset_name):
        super(WoTEnv, self).__init__()
        self.data, self.labels = load_dataset(dataset_name)
        self.num_samples, self.state_dim = self.data.shape
        self.action_dim = 2  # benign or malicious
        self.current_idx = 0

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.action_dim)

    def reset(self):
        self.current_idx = 0
        return self.data[self.current_idx]

    def step(self, action):
        label = self.labels[self.current_idx]
        reward = 1 if action == label else -1
        done = self.current_idx >= self.num_samples - 1
        self.current_idx += 1
        next_state = self.data[self.current_idx] if not done else np.zeros(self.state_dim)
        info = {}
        return next_state, reward, done, info

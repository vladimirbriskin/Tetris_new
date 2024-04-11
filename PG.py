import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

class PG_Agent:
    def __init__(self, input_dim, output_dim, learning_rate=0.001):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.policy_network = self.build_network()
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)

    def build_network(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, self.output_dim),
            torch.nn.Softmax(dim=-1)
        )
        return model

    def select_action(self, state, next_steps):
        # state = state.view(1, -1)
        action_probs = self.policy_network(state)
        action_distribution = torch.distributions.Categorical(action_probs)
        action = action_distribution.sample()
        if action.dim() == 0:
            action = action.unsqueeze(0)
        return (action.item(), 0), action_distribution.log_prob(action)

    def calculate_returns(self, rewards, discount_factor, normalize = True):
        returns = []
        R = 0

        for r in reversed(rewards):
            R = r + R * discount_factor
            returns.insert(0, R)

        returns = torch.tensor(returns)

        if normalize:
            returns = (returns - returns.mean()) / returns.std()

        return returns

    def update_policy(self, returns, log_prob_actions):
        returns = returns.detach()
        loss = - (returns * log_prob_actions).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save(self, filepath):
        torch.save(self.policy_network.state_dict(), filepath)

    def load(self, filepath):
        self.policy_network.load_state_dict(torch.load(filepath, map_location=self.device))
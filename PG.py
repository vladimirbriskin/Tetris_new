import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

class PG_Agent:
    def __init__(self, input_dim, output_dim, learning_rate=0.001, device=torch.device("cpu")):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.device = device
        self.policy_network = self.build_network()
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)

    def build_network(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, 128),
            torch.nn.Dropout(p=0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(128, self.output_dim)
        )
        return model.to(self.device)


    def select_action(self, next_steps):
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states).to(self.device)

        state_values = self.policy_network(next_states)
        # print(state_values)
        action_probs = F.softmax(state_values.squeeze(-1), dim=0)
        # print(action_probs)
        action_distribution = torch.distributions.Categorical(action_probs)
        actionIndex = action_distribution.sample()
        chosen_action = next_actions[actionIndex.item()]

        log_prob = action_distribution.log_prob(actionIndex)
        return chosen_action, log_prob.unsqueeze(0), actionIndex.item()

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
        returns = returns.detach().to(self.device)
        loss = - (returns * log_prob_actions).sum()
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.item()

    def save(self, filepath):
        torch.save(self.policy_network.state_dict(), filepath)

    def load(self, filepath):
        self.policy_network.load_state_dict(torch.load(filepath, map_location=self.device))
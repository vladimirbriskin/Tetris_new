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

    # def update_policy(self, rewards, log_probs):
    #     discounted_rewards = self.discount_rewards(rewards)
    #     policy_loss = []
    #     for log_prob, reward in zip(log_probs, discounted_rewards):
    #         policy_loss.append(-log_prob * reward)
    #     policy_loss = torch.cat(policy_loss).sum()

    #     self.optimizer.zero_grad()
    #     policy_loss.backward()
    #     self.optimizer.step()

    # def discount_rewards(self, rewards, gamma=0.99):
    #     discounted_rewards = np.zeros_like(rewards)
    #     running_add = 0
    #     for t in reversed(range(len(rewards))):
    #         running_add = running_add * gamma + rewards[t]
    #         discounted_rewards[t] = running_add
    #     discounted_rewards -= np.mean(discounted_rewards)
    #     discounted_rewards /= np.std(discounted_rewards)
    #     return discounted_rewards

# class PolicyGradient(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(PolicyGradient, self).__init__()
#         self.affine1 = nn.Linear(input_dim, 128)
#         self.affine2 = nn.Linear(128, output_dim)

#         self.saved_log_probs = []
#         self.rewards = []

#     def forward(self, x):
#         x = F.relu(self.affine1(x))
#         action_scores = self.affine2(x)
#         return F.softmax(action_scores, dim=1)

# class PG_Agent:
#     def __init__(self, input_dim, output_dim, device):
#         self.model = PolicyGradient(input_dim, output_dim).to(device)
#         self.learning_rate = 0.001
#         self.gamma = 0.99
#         self.eps = np.finfo(np.float32).eps.item()
#         self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
#         self.rewards = []
#         self.actions = []
#         self.device = device

#     def select_action(self, state):
#         probs = self.model(state).detach().numpy()
#         action = np.random.choice(self.output_dim, p=probs)
#         return action

#     def finish_episode(self):
#         R = 0
#         policy_loss = []
#         rewards = []
#         for r in self.model.rewards[::-1]:
#             R = r + self.gamma * R
#             rewards.insert(0, R)
#         rewards = torch.tensor(rewards)
#         rewards = (rewards - rewards.mean()) / (rewards.std() + self.eps)
#         # for log_prob, reward in zip(self.model.saved_log_probs, rewards):
#         #     policy_loss.append(-log_prob * reward)
#         self.optimizer.zero_grad()
#         policy_loss = torch.cat(policy_loss).sum()
#         policy_loss.backward()
#         self.optimizer.step()
#         del self.model.rewards[:]
#         del self.model.saved_log_probs[:]
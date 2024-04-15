"""
Based on https://github.com/bentrevett/pytorch-rl/, with modifications
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout = 0.5):
        super().__init__()

        self.fc_1 = nn.Linear(input_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc_1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc_2(x)
        return x

class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()

        self.actor = actor
        self.critic = critic

    def forward(self, state):

        action_pred = self.actor(state)
        value_pred = self.critic(state)

        return action_pred, value_pred

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)

class PPO_Agent:
    def __init__(self, input_dim, output_dim, env, learning_rate=0.01, clip_ratio=0.2, gamma=0.99, gae_lambda=0.95, ppo_steps=5):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = 128
        self.learning_rate = learning_rate
        self.clip_ratio = clip_ratio
        self.ppo_steps = ppo_steps
        self.gamma = gamma
        self.policy_network = self.build_network()
        self.policy_network.apply(init_weights)
        self.env = env
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)

    def build_network(self):
        actor = MLP(self.input_dim, self.hidden_dim, self.output_dim)
        critic = MLP(self.input_dim, self.hidden_dim, 1)
        return ActorCritic(actor, critic)

    def train(self):
        self.policy_network.train()

        states = []
        actions = []
        log_prob_actions = []
        values = []
        rewards = []
        done = False
        episode_reward = 0

        state = self.env.reset()

        while not done:

            next_steps = self.env.get_next_states()
            next_actions, next_states = zip(*next_steps.items())
            next_states = torch.stack(next_states)

            #append state here, not after we get the next state from env.step()
            # states.append(state)
            # print(state)
            states.append(torch.FloatTensor(self.env.get_current_board_state()).view(-1))

            action_pred, value_pred = self.policy_network(next_states)

            action_prob = F.softmax(action_pred.squeeze(-1), dim=0)

            dist = distributions.Categorical(action_prob)

            actionIndex = dist.sample()
            action = next_actions[actionIndex]
            # next_state = next_states[actionIndex, :]

            reward, done = self.env.step(action, render=False)

            actions.append(torch.tensor(action, dtype=torch.float32))
            log_prob_actions.append(dist.log_prob(actionIndex).unsqueeze(0))
            values.append(value_pred[actionIndex])
            rewards.append(reward)

            episode_reward += reward

        states = torch.stack(states)
        actions = torch.stack(actions)
        log_prob_actions = torch.cat(log_prob_actions)
        values = torch.cat(values).squeeze(-1)
        returns = self.calculate_returns(rewards, self.gamma)
        advantages = self.calculate_advantages(returns, values)

        policy_loss, value_loss = self.update_policy(self.policy_network, states, actions, log_prob_actions, advantages, returns, self.optimizer, self.ppo_steps, self.clip_ratio)

        return policy_loss, value_loss, episode_reward

    def evaluate(self):

        self.policy_network.eval()

        done = False
        episode_reward = 0

        state = self.env.reset()

        while not done:
            next_steps = self.env.get_next_states()
            next_actions, next_states = zip(*next_steps.items())
            next_states = torch.stack(next_states)

            with torch.no_grad():

                action_pred, _ = self.policy_network(next_states)

                action_prob = F.softmax(action_pred.squeeze(-1), dim=0)


            actionIndex = torch.argmax(action_prob, dim = -1)
            action = next_actions[actionIndex]
            # next_state = next_states[actionIndex, :]

            reward, done = self.env.step(action, render=False)
            episode_reward += reward

        return episode_reward

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

    def calculate_advantages(self, returns, values, normalize = True):

        advantages = returns - values

        if normalize:

            advantages = (advantages - advantages.mean()) / advantages.std()

        return advantages

    def update_policy(self, policy, states, actions, log_prob_actions, advantages, returns, optimizer, ppo_steps, ppo_clip):
        total_policy_loss = 0
        total_value_loss = 0

        advantages = advantages.detach()
        log_prob_actions = log_prob_actions.detach()
        actions = actions.detach()

        for _ in range(ppo_steps):
            # Re-compute action predictions and value predictions for the states
            action_pred, value_pred = policy(states)
            value_pred = value_pred.squeeze(-1)  # Ensure value predictions are correctly shaped

            action_probs = F.softmax(action_pred.squeeze(-1), dim=0)
            dist = torch.distributions.Categorical(action_probs)

            actionIndex = dist.sample()
            chosen_action = actions[actionIndex.item()]
            new_log_prob_actions = dist.log_prob(actionIndex)

            policy_ratio = torch.exp(new_log_prob_actions - log_prob_actions)

            # Calculate the two sides of the clipped objective function
            policy_loss_1 = policy_ratio * advantages
            policy_loss_2 = torch.clamp(policy_ratio, 1.0 - ppo_clip, 1.0 + ppo_clip) * advantages

            # Aggregate the losses
            policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()  # Mean instead of sum for normalization
            value_loss = F.smooth_l1_loss(returns, value_pred).mean()  # Mean squared error for value loss

            optimizer.zero_grad()
            (policy_loss + value_loss).backward()
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

        return total_policy_loss / ppo_steps, total_value_loss / ppo_steps

    def save(self, filepath):
        torch.save(self.policy_network.state_dict(), filepath)

    def load(self, filepath):
        self.policy_network.load_state_dict(torch.load(filepath, map_location=self.device))
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import copy
import numpy as np
import random
from collections import deque, namedtuple
from torch.distributions import Categorical
from base_algorithm import BaseAlgorithm
import sys

# class ReplayBuffer:
#     def __init__(self, buffer_size, batch_size, device):
#         self.device = device
#         self.memory = deque(maxlen=buffer_size)  
#         self.batch_size = batch_size
#         self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
#     def add(self, state, action, reward, next_state, done):
#         e = self.experience(state, action, reward, next_state, done)
#         self.memory.append(e)
    
#     def sample(self):
#         experiences = random.sample(self.memory, k=self.batch_size)
#         states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(self.device)
#         actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
#         rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
#         next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(self.device)
#         dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
#         return (states, actions, rewards, next_states, dones)

#     def __len__(self):
#         return len(self.memory)

# def hidden_init(layer):
#     fan_in = layer.weight.data.size()[0]
#     lim = 1. / np.sqrt(fan_in)
#     return (-lim, lim)

# class Actor(nn.Module):
#     def __init__(self, state_size, action_size, hidden_size=32):
#         super(Actor, self).__init__()
#         self.fc1 = nn.Linear(state_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, hidden_size)
#         self.fc3 = nn.Linear(hidden_size, action_size)
#         # self.softmax = nn.Softmax(dim=-1)

#     def forward(self, state):
#         x = F.relu(self.fc1(state))
#         x = F.relu(self.fc2(x))
#         # action_probs = self.softmax(self.fc3(x))
#         # return action_probs
#         x = self.fc3(x)
#         return x

#     def evaluate(self, state, epsilon=1e-6):
#         action_probs = self.forward(state)
#         dist = Categorical(action_probs)
#         action = dist.sample().to(state.device)
#         # Have to deal with situation of 0.0 probabilities because we can't do log 0
#         z = action_probs == 0.0
#         z = z.float() * 1e-8
#         log_action_probabilities = torch.log(action_probs + z)
#         return action.detach().cpu(), action_probs, log_action_probabilities        
    
#     def get_action(self, state):
#         action_probs = self.forward(state)
#         dist = Categorical(action_probs)
#         action = dist.sample().to(state.device)
#         # Have to deal with situation of 0.0 probabilities because we can't do log 0
#         z = action_probs == 0.0
#         z = z.float() * 1e-8
#         log_action_probabilities = torch.log(action_probs + z)
#         return action.detach().cpu(), action_probs, log_action_probabilities
    
#     def get_det_action(self, state):
#         action_probs = self.forward(state)
#         dist = Categorical(action_probs)
#         print(action_probs)
#         sys.exit()
#         action = dist.sample().to(state.device)
#         return action.detach().cpu()


# class Critic(nn.Module):
#     def __init__(self, state_size, action_size, hidden_size=32, seed=1):
#         super(Critic, self).__init__()
#         self.seed = torch.manual_seed(seed)
#         self.fc1 = nn.Linear(state_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, hidden_size)
#         self.fc3 = nn.Linear(hidden_size, action_size)
#         self.reset_parameters()

#     def reset_parameters(self):
#         self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
#         self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
#         self.fc3.weight.data.uniform_(-3e-3, 3e-3)

#     def forward(self, state):
#         state = state.view(state.size(0), -1)
#         x = F.relu(self.fc1(state))
#         x = F.relu(self.fc2(x))
#         return self.fc3(x)


# class SAC_Agent(BaseAlgorithm):
#     def __init__(self, state_size, action_size, device):
#         super(SAC_Agent, self).__init__()
#         self.state_size = state_size
#         self.action_size = action_size
#         self.device = device
#         self.gamma = 0.99
#         self.tau = 1e-2
#         hidden_size = 256
#         learning_rate = 5e-4
#         self.clip_grad_param = 1
#         self.target_entropy = -action_size  # -dim(A)
#         self.log_alpha = torch.tensor([0.0], requires_grad=True)
#         self.alpha = self.log_alpha.exp().detach()
#         self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=learning_rate) 
        
#         # Actor Network 
#         self.actor_local = Actor(state_size, action_size, hidden_size).to(device)
#         self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=learning_rate)     
        
#         # Critic Network (w/ Target Network)
#         self.critic1 = Critic(state_size, action_size, hidden_size, 2).to(device)
#         self.critic2 = Critic(state_size, action_size, hidden_size, 1).to(device)
        
#         assert self.critic1.parameters() != self.critic2.parameters()
        
#         self.critic1_target = Critic(state_size, action_size, hidden_size).to(device)
#         self.critic1_target.load_state_dict(self.critic1.state_dict())

#         self.critic2_target = Critic(state_size, action_size, hidden_size).to(device)
#         self.critic2_target.load_state_dict(self.critic2.state_dict())

#         self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
#         self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate) 
    
#     def select_action(self, state):
#         next_actions, next_states = zip(*state.items())
#         next_states = torch.stack(next_states)
#         if torch.cuda.is_available():
#             next_states = next_states.cuda()
#         with torch.no_grad():
#             predictions = self.actor_local.get_det_action(next_states)
#         index = torch.argmax(predictions).item()
#         next_state = next_states[index, :]
#         action = next_actions[index]
#         return action, next_state

#     def calc_policy_loss(self, states, alpha):
#         _, action_probs, log_pis = self.actor_local.evaluate(states)
#         q1 = self.critic1(states)   
#         q2 = self.critic2(states)
#         min_Q = torch.min(q1,q2)
#         actor_loss = (action_probs * (alpha * log_pis - min_Q )).sum(1).mean()
#         log_action_pi = torch.sum(log_pis * action_probs, dim=1)
#         return actor_loss, log_action_pi
    
#     def optimize_model(self, experiences, gamma, d=1):
#         states, actions, rewards, next_states, dones = experiences
#         # ---------------------------- update actor ---------------------------- #
#         current_alpha = copy.deepcopy(self.alpha)
#         actor_loss, log_pis = self.calc_policy_loss(states, current_alpha.to(self.device))
#         self.actor_optimizer.zero_grad()
#         actor_loss.backward()
#         self.actor_optimizer.step()
#         # Compute alpha loss
#         alpha_loss = - (self.log_alpha.exp() * (log_pis.cpu() + self.target_entropy).detach().cpu()).mean()
#         self.alpha_optimizer.zero_grad()
#         alpha_loss.backward()
#         self.alpha_optimizer.step()
#         self.alpha = self.log_alpha.exp().detach()
#         # ---------------------------- update critic ---------------------------- #
#         # Get predicted next-state actions and Q values from target models
#         with torch.no_grad():
#             _, action_probs, log_pis = self.actor_local.evaluate(next_states)
#             Q_target1_next = self.critic1_target(next_states)
#             Q_target2_next = self.critic2_target(next_states)
#             Q_target_next = action_probs * (torch.min(Q_target1_next, Q_target2_next) - self.alpha.to(self.device) * log_pis)

#             # Compute Q targets for current states (y_i)
#             Q_targets = rewards + (gamma * (1 - dones) * Q_target_next.sum(dim=1).unsqueeze(-1)) 

#         # Compute critic loss
#         q1 = self.critic1(states).gather(1, actions.long())
#         q2 = self.critic2(states).gather(1, actions.long())
        
#         critic1_loss = 0.5 * F.mse_loss(q1, Q_targets)
#         critic2_loss = 0.5 * F.mse_loss(q2, Q_targets)

#         # Update critics
#         # critic 1
#         self.critic1_optimizer.zero_grad()
#         critic1_loss.backward(retain_graph=True)
#         clip_grad_norm_(self.critic1.parameters(), self.clip_grad_param)
#         self.critic1_optimizer.step()
#         # critic 2
#         self.critic2_optimizer.zero_grad()
#         critic2_loss.backward()
#         clip_grad_norm_(self.critic2.parameters(), self.clip_grad_param)
#         self.critic2_optimizer.step()

#         # ----------------------- update target networks ----------------------- #
#         self.soft_update(self.critic1, self.critic1_target)
#         self.soft_update(self.critic2, self.critic2_target)
        
#         return actor_loss.item(), alpha_loss.item(), critic1_loss.item(), critic2_loss.item(), current_alpha

#     def soft_update(self, local_model , target_model):
#         for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
#             target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

#     def save(self, filepath):
#         torch.save(self.actor_local.state_dict(), filepath)
    
#     def load(self, filepath):
#         self.actor_local.load_state_dict(torch.load(filepath, map_location=self.device))


# Define the neural network architecture for the actor and critic
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, action_indexes, rewards, next_states, dones = zip(*batch)
        return states, actions, action_indexes, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # Tanh activation for bounded action space
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the Soft Actor-Critic agent
class SAC_Agent(BaseAlgorithm):
    def __init__(self, state_dim, action_dim, device, hidden_size=256, gamma=0.99, alpha=0.2, tau=0.005, lr=3e-4):
        self.actor = Actor(state_dim, action_dim, hidden_size).to(device)
        self.actor_target = Actor(state_dim, action_dim, hidden_size).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic1 = Critic(state_dim, action_dim, hidden_size).to(device)
        self.critic1_target = Critic(state_dim, action_dim, hidden_size).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)

        self.critic2 = Critic(state_dim, action_dim, hidden_size).to(device)
        self.critic2_target = Critic(state_dim, action_dim, hidden_size).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)

        self.log_alpha = torch.tensor(np.log(alpha)).to(device)
        self.log_alpha.requires_grad = True
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.target_entropy = -action_dim
        self.device = device

    def select_action(self, state):
        next_actions, next_states = zip(*state.items())
        next_states = torch.stack(next_states)
        if torch.cuda.is_available():
            next_states = next_states.cuda()
        # with torch.no_grad():
            # state = torch.FloatTensor(state).to(self.device)
            # action = self.actor(state).cpu().numpy()
        self.actor.eval()
        with torch.no_grad():
            predictions = self.actor(next_states)[:, 0]
        self.actor.train()
        index = torch.argmax(predictions).item()
        next_state = next_states[index, :]
        action = next_actions[index]
        action_index = index
        return action, next_state, action_index

    def optimize_model(self, replay_buffer, batch_size=128):
        state_batch, action_batch, action_index_batch, reward_batch, next_state_batch, done_batch = replay_buffer.sample(batch_size)

        
        # state_batch = torch.FloatTensor(state_batch).to(self.device)
        # action_batch = torch.FloatTensor(action_batch).to(self.device)
        # reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        # next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        # done_batch = torch.FloatTensor(1 - done_batch).to(self.device)
        state_batch = torch.stack(tuple(state for state in state_batch)).to(self.device)
        action_batch = torch.stack(tuple(action[2:4] for action in action_batch)).to(self.device)
        action_index_batch = torch.stack(tuple(action_index for action_index in action_index_batch)).to(self.device)
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None]).to(self.device)
        next_state_batch = torch.stack(tuple(next_state for next_state in next_state_batch)).to(self.device)
        done_batch = torch.from_numpy(1 - np.array(done_batch, dtype=np.float32)[:, None]).to(self.device)

        # Update critic networks
        target_Q = reward_batch + self.gamma * torch.min(
            self.critic1_target(next_state_batch, self.actor_target(next_state_batch)),
            self.critic2_target(next_state_batch, self.actor_target(next_state_batch))
        ) * done_batch

        # current_Q1 = self.critic1(state_batch, action_index_batch)
        # current_Q2 = self.critic2(state_batch, action_index_batch)

        current_Q1 = self.critic1(state_batch, action_batch)
        current_Q2 = self.critic2(state_batch, action_batch)

        critic1_loss = nn.functional.mse_loss(current_Q1, target_Q.detach())
        critic2_loss = nn.functional.mse_loss(current_Q2, target_Q.detach())

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Update actor network
        actor_loss = (self.alpha * self.actor(state_batch)).mean() - torch.min(
            self.critic1(state_batch, self.actor(state_batch)),
            self.critic2(state_batch, self.actor(state_batch))
        ).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks with Polyak averaging
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Update entropy temperature parameter
        alpha_loss = -(self.alpha * (self.target_entropy - (self.alpha * self.actor(state_batch).log()).mean())).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        return actor_loss

    def save(self, filepath):
        torch.save(self.actor.state_dict(), filepath)
    
    def load(self, filepath):
        self.actor.load_state_dict(torch.load(filepath, map_location=self.device))
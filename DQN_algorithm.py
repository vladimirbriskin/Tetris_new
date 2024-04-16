from DQN_network import DeepQNetwork
from base_algorithm import BaseAlgorithm
from random import *
import torch
from random import random
import numpy as np
from collections import deque

class DQNAlgorithm(BaseAlgorithm):
    def __init__(self, args):
        self.model = DeepQNetwork().to(args['device'])
        self.target_model = DeepQNetwork().to(args['device'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args['lr'])
        self.criterion = torch.nn.MSELoss()
        self.device = args['device']
        self.replay_memory_size = args['replay_memory_size']
        self.replay_memory = deque(maxlen=args['replay_memory_size'])
        self.gamma = args['gamma']
        self.final_epsilon = args['final_epsilon']
        self.num_decay_epochs = args['num_decay_epochs']
        self.initial_epsilon = args['initial_epsilon']
        self.batch_size = args['batch_size']

    def select_action(self, state, epoch):
        # Exploration or exploitation
        epsilon = self.final_epsilon + (max(self.num_decay_epochs - epoch, 0) * (
                self.initial_epsilon - self.final_epsilon) / self.num_decay_epochs)
        u = random()
        random_action = u <= epsilon
        next_actions, next_states = zip(*state.items())
        next_states = torch.stack(next_states)
        # if torch.cuda.is_available():
        #     next_states = next_states.cuda()
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(next_states)[:, 0]
        self.model.train()
        if random_action:
            index = randint(0, len(state) - 1)
        else:
            index = torch.argmax(predictions).item()
        next_state = next_states[index, :]
        action = next_actions[index]
        return action, next_state

    def select_max_action(self,state):
        next_states = torch.stack(state)
        next_states = next_states.to(self.device)
        self.target_model.eval()
        with torch.no_grad():
            predictions = self.target_model(next_states)[:, 0] # DQN
        self.target_model.train()
        index = torch.argmax(predictions).item()
        return next_states[index,:]



    def add_replay(self, state, action, reward, next_state,next_next_state, done):
        self.replay_memory.append([state, action, reward, next_state,next_next_state, done])

    def optimize_model(self):
        if len(self.replay_memory) < self.replay_memory_size / 10:
            return None  # Not enough samples to optimize model

        batch = sample(self.replay_memory, min(len(self.replay_memory), self.batch_size))
        state_batch, action_batch, reward_batch, next_state_batch,next_next_state_batch, done_batch = zip(*batch)
        state_batch = torch.stack(tuple(state for state in state_batch))

        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.stack(tuple(state for state in next_state_batch))
        next_next_state_batch = torch.stack(tuple(state for state in next_next_state_batch))

        q_values = self.model(next_state_batch)
        self.target_model.eval()
        with torch.no_grad():
            next_prediction_batch = self.target_model(next_next_state_batch)
        self.target_model.train()

        y_batch = torch.cat(
            tuple(reward if done else reward + self.gamma * prediction for reward, done, prediction in
                  zip(reward_batch, done_batch, next_prediction_batch)))[:, None]

        self.optimizer.zero_grad()
        loss = self.criterion(q_values, y_batch)
        loss.backward()
        for param in self.model.parameters():  
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.item()

    def test_select_action(self,next_steps):
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)
        predictions = self.model(next_states)[:, 0]
        index = torch.argmax(predictions).item()
        action = next_actions[index]
        return action

    def save(self, filepath):
        torch.save(self.model.state_dict(), filepath)

    def load(self, filepath):
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))

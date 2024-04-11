import os
import shutil
import torch
import logging
from dqn_algorithm import DQNAlgorithm
from SAC import SAC_Agent, ReplayBuffer
from PG import PG_Agent
from tetris import Tetris
import yaml
import sys
import argparse
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="DQN")  # Changed default path
    args = parser.parse_args()
    return args

def setup_logging(log_path):
    if os.path.isdir(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path, exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_path, 'training.log'),
                        level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

def create_directories(opt):
    """Create directories specified in the options."""
    os.makedirs(opt['log_path'], exist_ok=True)
    os.makedirs(opt['saved_path'], exist_ok=True)

def collect_random(env, dataset, num_samples, agent):
    state = env.reset()
    for _ in range(num_samples):
        next_steps = env.get_next_states()
        action, next_state, action_index = agent.select_action(next_steps)
        reward, done = env.step(action, render=False)
        dataset.add((state, torch.tensor(action), torch.tensor(action_index).unsqueeze(dim=-1), reward, next_state, done))
        state = next_state
        if done:
            state = env.reset()

def train(opt, model_name):
    torch.manual_seed(123)
    setup_logging(opt['log_path'])  # Setup logging

    env = Tetris(width=opt['width'], height=opt['height'], block_size=opt['block_size'])
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print("model: {}".format(model_name))
    if model_name == "DQN":
        agent = DQNAlgorithm(opt)
        state = env.reset()
        epoch = 0
        while epoch < opt['num_epochs'] and env.score < opt['max_score']:
            next_steps = env.get_next_states()

            action, next_state = agent.select_action(next_steps, epoch)
            reward, done = env.step(action, render=False)
            print(action)
            agent.add_replay(state, reward, next_state, done)

            if done:
                final_score = env.score
                final_tetrominoes = env.tetrominoes
                final_cleared_lines = env.cleared_lines
                state = env.reset()
            else:
                state = next_state
                continue

            loss = agent.optimize_model()

            if loss is not None:  # Log loss if optimization occurred
                logging.info(f"Epoch: {epoch}, Loss: {loss}, Score: {final_score}, Tetrominoes: {final_tetrominoes}, Cleared lines: {final_cleared_lines}")

            epoch += 1
            if (epoch) % opt['target_update'] == 0:  # update the target network
                agent.target_model.load_state_dict(agent.model.state_dict())
            if epoch > 0 and epoch % opt['save_interval'] == 0:
                agent.save(f"{opt['saved_path']}/tetris_{epoch}.pth")

        agent.save(f"{opt['saved_path']}/tetris_final.pth")
    elif model_name == "SAC":
        state = env.reset()
        batch_size = opt['batch_size']
        buffer = ReplayBuffer(opt['replay_memory_size'])
        agent = SAC_Agent(state_dim=state.shape[0],
                          action_dim=2,
                          device=device,
                          hidden_size=opt['hidden_size'],
                          gamma=opt['gamma'],
                          alpha=opt['alpha'],
                          tau=opt['tau'],
                          lr=opt['lr'])
        collect_random(env=env, dataset=buffer, num_samples=10000, agent=agent)
        for epoch in range(opt['num_epochs']):
            done = False
            losses = []
            while not done:
                next_steps = env.get_next_states()
                action, next_state, action_index = agent.select_action(next_steps)
                reward, done = env.step(action, render=False)
                buffer.add((state, torch.tensor(action), torch.tensor(action_index).unsqueeze(dim=-1), reward, next_state, done))
                loss = agent.optimize_model(buffer, batch_size=batch_size)
                losses.append(loss.cpu().detach().numpy())
                state = next_state
            final_score = env.score
            final_tetrominoes = env.tetrominoes
            final_cleared_lines = env.cleared_lines
            state = env.reset()
            average_loss = np.mean(losses) if losses else 0
            logging.info(f"Epoch: {epoch}, Loss: {average_loss}, Score: {final_score}, Tetrominoes: {final_tetrominoes}, Cleared lines: {final_cleared_lines}")
            print(f"Epoch: {epoch}, Loss: {average_loss}, Score: {final_score}, Tetrominoes: {final_tetrominoes}, Cleared lines: {final_cleared_lines}")
            if epoch > 0 and epoch % opt['save_interval'] == 0:
                agent.save(f"{opt['saved_path']}/tetris_{epoch}.pth")
        agent.save(f"{opt['saved_path']}/tetris_final.pth")

    elif model_name == "PG": # Policy Gradient
        state = env.reset()
        state_dim = state.shape[0]
        action_dim = 2
        agent = PG_Agent(input_dim=state_dim, output_dim=action_dim, learning_rate = opt['lr'])
        agent.policy_network.train()
        discount_factor = opt['gamma']
        epoch = 0

        while epoch < opt['num_epochs'] and env.score < opt['max_score']:
            state = env.reset()
            done = False
            log_probs = []
            rewards = []
            episode_reward = 0
            state = torch.FloatTensor(state).unsqueeze(0)

            while not done:
                next_steps = env.get_next_states()

                action, log_prob_action = agent.select_action(state, next_steps)
                reward, done = env.step(action, render=False)
                state = env.get_state_properties(env.board)

                log_probs.append(log_prob_action)
                rewards.append(reward)
                episode_reward += reward

            log_probs = torch.cat(log_probs)
            rewards_cal = agent.calculate_returns(rewards, discount_factor)
            loss = agent.update_policy(rewards_cal, log_probs)

            epoch += 1

            # Logging
            final_score = env.score
            if epoch % opt['log_interval'] == 0:
                logging.info(f"Epoch: {epoch}, Score: {final_score}, Loss: {loss}, Rewards: {rewards_cal}")

            if epoch > 0 and epoch % opt['save_interval'] == 0:
                agent.save(f"{opt['saved_path']}/tetris_pg_{epoch}.pth")

        agent.save(f"{opt['saved_path']}/tetris_pg_final.pth")

if __name__ == "__main__":
    args = get_args()
    model_name = args.model
    config_file_path = 'config.yaml'
    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)
    for model_config in config['models']:
        if model_name == model_config['model_name']:
            parameters = model_config['parameters']
            create_directories(parameters)
            train(parameters, model_name)

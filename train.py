import argparse
import os
import shutil
import torch
import logging  # Import logging
from dqn_algorithm import DQNAlgorithm
from tetris import Tetris
import yaml

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

def train(opt):
    torch.manual_seed(123)
    setup_logging(opt['log_path'])  # Setup logging

    env = Tetris(width=opt['width'], height=opt['height'], block_size=opt['block_size'])
    agent = DQNAlgorithm(opt)
    state = env.reset()
    epoch = 0
    while epoch < opt['num_epochs'] and env.score < opt['max_score']:
        next_steps = env.get_next_states()

        action, next_state = agent.select_action(next_steps, epoch)
        reward, done = env.step(action, render=False)
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

        if epoch > 0 and epoch % opt['save_interval'] == 0:
            agent.save(f"{opt['saved_path']}/tetris_{epoch}.pth")

    agent.save(f"{opt['saved_path']}/tetris_final.pth")


if __name__ == "__main__":
    config_file_path = 'config.yaml'
    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)

    for model_config in config['models']:
        model_name = model_config['model_name']
        parameters = model_config['parameters']

        # Initialize parameters for DQN
        if model_name == 'DQN':
            # Initialize your DQN model here with the parameters
            opt = parameters
            create_directories(opt)
            train(opt)
            
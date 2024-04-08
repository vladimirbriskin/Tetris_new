import os
import shutil
import torch
import logging  
from dqn_algorithm import DQNAlgorithm
from SAC import SAC_Agent, ReplayBuffer
from tetris import Tetris
import yaml
import sys

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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("model: {}".format(model_name))
    if model_name == "DQN":
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
    elif model_name == "SAC":
        state = env.reset()
        batch_size = opt['batch_size']
        buffer = ReplayBuffer(opt['replay_memory_size'])
        agent = SAC_Agent(state_dim=state.shape[0], action_dim=1, device=device)
        collect_random(env=env, dataset=buffer, num_samples=10000, agent=agent)
        epoch = 0
        while epoch < opt['num_epochs'] and env.score < opt['max_score']:
            next_steps = env.get_next_states()
            action, next_state, action_index = agent.select_action(next_steps)
            reward, done = env.step(action, render=False)
            
            buffer.add((state, torch.tensor(action), torch.tensor(action_index).unsqueeze(dim=-1), reward, next_state, done))
            loss = agent.optimize_model(buffer, batch_size=batch_size)
            if done:
                final_score = env.score
                final_tetrominoes = env.tetrominoes
                final_cleared_lines = env.cleared_lines
                state = env.reset()
            else:
                state = next_state
                continue
            if loss is not None:  # Log loss if optimization occurred
                logging.info(f"Epoch: {epoch}, Loss: {loss}, Score: {final_score}, Tetrominoes: {final_tetrominoes}, Cleared lines: {final_cleared_lines}")
                # print(f"Epoch: {epoch}, Loss: {loss}, Score: {final_score}, Tetrominoes: {final_tetrominoes}, Cleared lines: {final_cleared_lines}")
            epoch += 1
            if epoch > 0 and epoch % opt['save_interval'] == 0:
                agent.save(f"{opt['saved_path']}/tetris_{epoch}.pth")
        agent.save(f"{opt['saved_path']}/tetris_final.pth")  

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
            

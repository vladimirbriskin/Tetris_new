from tetris import Tetris
from DQN_algorithm import DQNAlgorithm
from DDQN_algorithm import DDQNAlgorithm
from NoisyDQN_algorithm import NoisyDQNAlgorithm
import yaml
from SAC import SAC_Agent, ReplayBuffer
import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="DQN")  # Changed default path
    args = parser.parse_args()
    return args

def test(opt, agent, model_name, num_epochs=10):
    # Initialize the Tetris environment with the specified configuration
    env = Tetris(width=opt['width'], height=opt['height'], block_size=opt['block_size'])

    # Load the state_dict into your model
    agent.load(f"{opt['saved_path']}/tetris_final.pth")

    # Variables to keep track of scores, tetrominoes, and lines cleared
    total_score = 0
    total_tetrominoes = 0
    total_cleared_lines = 0
    # Run the model for a specified number of epochs
    if model_name == "DQN":
        agent.model.eval()
        for epoch in range(num_epochs):
            state = env.reset()
            done = False
            
            while not done:
                next_steps = env.get_next_states()
                action = agent.test_select_action(next_steps)
                _, done = env.step(action, render=False)

            # Accumulate totals
            total_score += env.score
            total_tetrominoes += env.tetrominoes
            total_cleared_lines += env.cleared_lines
            
            print(f"Epoch {epoch + 1}: Score = {env.score}, Tetrominoes = {env.tetrominoes}, Cleared lines = {env.cleared_lines}")
    elif model_name == "DDQN":
        agent.model.eval()
        for epoch in range(num_epochs):
            state = env.reset()
            done = False
            
            while not done:
                next_steps = env.get_next_states()
                action = agent.test_select_action(next_steps)
                _, done = env.step(action, render=False)

            # Accumulate totals
            total_score += env.score
            total_tetrominoes += env.tetrominoes
            total_cleared_lines += env.cleared_lines
            
            print(f"Epoch {epoch + 1}: Score = {env.score}, Tetrominoes = {env.tetrominoes}, Cleared lines = {env.cleared_lines}")

    elif model_name == "NoisyDQN":
        agent.model.eval()
        for epoch in range(num_epochs):
            state = env.reset()
            done = False
            
            while not done:
                next_steps = env.get_next_states()
                action = agent.test_select_action(next_steps)
                _, done = env.step(action, render=False)

            # Accumulate totals
            total_score += env.score
            total_tetrominoes += env.tetrominoes
            total_cleared_lines += env.cleared_lines
            
            print(f"Epoch {epoch + 1}: Score = {env.score}, Tetrominoes = {env.tetrominoes}, Cleared lines = {env.cleared_lines}")


    elif model_name == "SAC":
        agent.actor.eval()
        for epoch in range(num_epochs):
            state = env.reset()
            done = False
            
            while not done:
                next_steps = env.get_next_states()
                action, next_state, action_index = agent.select_action(next_steps)
                reward, done = env.step(action, render=True)

            # Accumulate totals
            total_score += env.score
            total_tetrominoes += env.tetrominoes
            total_cleared_lines += env.cleared_lines
            
            print(f"Epoch {epoch + 1}: Score = {env.score}, Tetrominoes = {env.tetrominoes}, Cleared lines = {env.cleared_lines}")

    # Calculate and print averages
    avg_score = total_score / num_epochs
    avg_tetrominoes = total_tetrominoes / num_epochs
    avg_cleared_lines = total_cleared_lines / num_epochs

    print(f"Average Score: {avg_score}, Average Tetrominoes: {avg_tetrominoes}, Average Cleared Lines: {avg_cleared_lines}")

if __name__ == "__main__":
    args = get_args()
    model_name = args.model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config_file_path = 'config.yaml'
    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)

    # Iterate through each model configuration and test
    for model_config in config['models']:
        if model_name == model_config['model_name']:
            parameters = model_config['parameters']
            if model_name == "DQN":
                agent = DQNAlgorithm(parameters)
            elif model_name == "DDQN":
                agent = DDQNAlgorithm(parameters)
            elif model_name == "NoisyDQN":
                agent = NoisyDQNAlgorithm(parameters)
            elif model_name == "SAC":
                agent = SAC_Agent(state_dim=4, 
                                  action_dim=2, 
                                  device=device, 
                                  hidden_size=parameters['hidden_size'], 
                                  gamma=parameters['gamma'], 
                                  alpha=parameters['alpha'], 
                                  tau=parameters['tau'], 
                                  lr=parameters['lr'])
            test(parameters, agent, model_name)


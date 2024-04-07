import torch
from tetris import Tetris
from dqn_algorithm import DQNAlgorithm
import yaml

def test(opt, num_epochs=10):
    # Initialize the Tetris environment with the specified configuration
    env = Tetris(width=opt['width'], height=opt['height'], block_size=opt['block_size'])
    
    # Initialize the agent with the options
    agent = DQNAlgorithm(opt)
    
    # Load the trained model and ensure it is in evaluation mode
    agent.load(f"{opt['saved_path']}/tetris_final.pth")
    agent.model.eval()

    # Variables to keep track of scores, tetrominoes, and lines cleared
    total_score = 0
    total_tetrominoes = 0
    total_cleared_lines = 0

    # Run the model for a specified number of epochs
    for epoch in range(num_epochs):
        state = env.reset()
        done = False
        
        while not done:
            next_steps = env.get_next_states()
            action, _ = agent.select_action(next_steps, 0)  # Epoch=0 to disable exploration
            _, done = env.step(action, render=False)  # Change render to True if you want to watch the game

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
    config_file_path = 'config.yaml'
    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)

    # Iterate through each model configuration and test
    for model_config in config['models']:
        model_name = model_config['model_name']
        if model_name == 'DQN':
            parameters = model_config['parameters']
            test(parameters)

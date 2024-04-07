import argparse
import os
import shutil
import torch
import logging  # Import logging
from dqn_algorithm import DQNAlgorithm
from tetris import Tetris

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")
    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--batch_size", type=int, default=512, help="The number of images per batch")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=1)
    parser.add_argument("--final_epsilon", type=float, default=1e-3)
    parser.add_argument("--num_decay_epochs", type=float, default=2000)
    parser.add_argument("--num_epochs", type=int, default=3000)
    parser.add_argument("--save_interval", type=int, default=200)
    parser.add_argument("--replay_memory_size", type=int, default=30000,
                        help="Number of epoches between testing phases")
    parser.add_argument("--log_path", type=str, default="logs")  # Changed default path
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--device", type=str, default="cpu")

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

def train(opt):
    torch.manual_seed(123)
    setup_logging(opt.log_path)  # Setup logging

    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)
    agent = DQNAlgorithm(opt)
    state = env.reset()
    epoch = 0
    while epoch < opt.num_epochs:
        next_steps = env.get_next_states()

        action, next_state = agent.select_action(next_steps, epoch)
        reward, done = env.step(action, render=True)
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

        if epoch > 0 and epoch % opt.save_interval == 0:
            agent.save(f"{opt.saved_path}/tetris_{epoch}.pth")

    agent.save(f"{opt.saved_path}/tetris_final.pth")


if __name__ == "__main__":
    opt = get_args()
    train(opt)

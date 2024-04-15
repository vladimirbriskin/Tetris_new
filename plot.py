import matplotlib.pyplot as plt
from datetime import datetime
import os
import argparse
import sys

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="DQN")  # Changed default path
    args = parser.parse_args()
    return args

def plot_and_save_metric(epochs, metric_name, metric_values, save_file):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metric_values)
    plt.title(f'{metric_name} over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.grid(True)
    plt.savefig(save_file)
    plt.close()

def plot_score(model_name):
    root = "./" + model_name
    training_log_file = os.path.join(root, "logs/training.log")
    epochs = []
    scores = []
    tetrominoes = []
    cleared_lines = []

    with open(training_log_file, 'r') as file:
        for line in file:
            if "INFO" in line:
                parts = line.split(",")
                epoch = int(parts[0].split(":")[-1].strip())
                score = int(parts[2].split(":")[1].strip())
                tetrominoe = int(parts[3].split(":")[1].strip())
                cleared_line = int(parts[4].split(":")[1].strip())
                epochs.append(epoch)
                scores.append(score)
                tetrominoes.append(tetrominoe)
                cleared_lines.append(cleared_line)
    plot_and_save_metric(epochs, 'Score', scores, os.path.join(root, 'score_plot.png'))
    plot_and_save_metric(epochs, 'Tetrominoes', tetrominoes, os.path.join(root, 'tetrominoes_plot.png'))
    plot_and_save_metric(epochs, 'Cleared Lines', cleared_lines, os.path.join(root, 'cleared_lines_plot.png'))

if __name__ == '__main__':
    args = get_args()
    model_name = args.model
    plot_score(model_name)
import os
import re
import numpy as np
import matplotlib.pyplot as plt

def extract_scores(log_file):
    scores = []
    with open(log_file, 'r') as file:
        for line in file:
            if "INFO" in line:
                parts = line.split(",")
                score = int(parts[2].split(":")[1].strip())
                scores.append(score)
    return scores

def save_plot(plt, file_name):
    plt.savefig(file_name)
    print(f"Plot saved as '{file_name}'")

def plot(log_folder):
    all_scores = []
    for file_name in os.listdir(log_folder):
        if file_name.endswith('.log'):
            log_file = os.path.join(log_folder, file_name)
            scores = extract_scores(log_file)
            all_scores.append(scores)

    all_scores = np.array(all_scores)
    mean_scores = np.mean(all_scores, axis=0)
    std_scores = np.std(all_scores, axis=0)
    max_scores = np.max(all_scores, axis=0)
    min_scores = np.min(all_scores, axis=0)
    epochs = range(1, len(mean_scores) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, mean_scores, color='#335BFF', label="mean")
    # plt.fill_between(epochs, mean_scores - std_scores, mean_scores + std_scores, alpha=.7, color='#33E3FF', label="mean±std")
    plt.fill_between(epochs, min_scores, max_scores, alpha=.7, color='#33E3FF')

    plt.xlabel('Epoch')
    plt.ylabel('Score')
    # plt.title(plot_title)
    plt.legend()
    plt.grid(True)
    save_plot(plt, os.path.join(log_folder, "result.png"))
    plt.show()

if __name__ == "__main__":
    plot(log_folder="./logs")


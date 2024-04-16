# Tetris_new

# Usage

To get started with this project, follow these steps:
### Step 1: Configure the Application
Before running the training or testing scripts, you must configure the application settings. To do this, edit the `config.yaml` file located in the root directory. Ensure you set the desired configurations for training and environment parameters.

### Step 2: Training the Agent
Once you have configured the settings, you can train the agent by executing the `train.py` script. Run the following command in your terminal:

```bash
python train.py --model DQN
```

This script will start the training process based on the parameters defined in `config.yaml`. Make sure you monitor the output for any errors or important information regarding the training progress.

### Step 3: Testing the Agent
After training the agent, you can evaluate its performance by running the `test.py` script. Execute the following command:

```bash
python test.py --model DQN
```

This will initiate the testing process, using the trained model to play Tetris. Observe the agent to ensure it performs as expected based on the training it received.

---

import os
import logging
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd

###############################################################################
# Setup Logging and Device
###############################################################################
os.makedirs("plots", exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

###############################################################################
# Q-Network
###############################################################################
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=(128, 128)):
        super(QNetwork, self).__init__()
        layers = []
        input_dim = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, action_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

###############################################################################
# Replay Buffer
###############################################################################
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

###############################################################################
# Single Transition Update Function (for naive config only)
###############################################################################
def update_single(policy_net, target_net, optimizer, gamma, transition):
    state, action, reward, next_state, done = transition

    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
    action_tensor = torch.LongTensor([action]).to(device)
    reward_tensor = torch.FloatTensor([reward]).to(device)
    done_tensor = torch.FloatTensor([float(done)]).to(device)

    q_values = policy_net(state_tensor).gather(1, action_tensor.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        if target_net is not None:
            next_q_values = target_net(next_state_tensor).max(1)[0]
        else:
            next_q_values = policy_net(next_state_tensor).max(1)[0]
        target_q_values = reward_tensor + gamma * next_q_values * (1 - done_tensor)

    loss = nn.MSELoss()(q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

###############################################################################
# Batch Update Function (for replay-buffer-based configs)
###############################################################################
def update_batch(policy_net, target_net, optimizer, gamma, batch):
    """
    Updates the policy_net using a batch of transitions from the replay buffer.
    Used when config=='only_tn', 'only_er', or 'tn_er'.
    """
    states, actions, rewards, next_states, dones = batch

    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).unsqueeze(1).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.FloatTensor(dones).to(device)
    
    q_values = policy_net(states).gather(1, actions)

    with torch.no_grad():
        if target_net is not None:
            next_q_values = target_net(next_states).max(1)[0]
        else:
            next_q_values = policy_net(next_states).max(1)[0]
        target_q_values = rewards + gamma * next_q_values * (1 - dones)

    loss = nn.MSELoss()(q_values.squeeze(), target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

###############################################################################
# Training Function
###############################################################################
def train_cartpole_dqn(
    config,
    total_timesteps=100000,
    batch_size=256,
    gamma=0.99,
    lr=5e-4,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=10000,
    buffer_capacity=10000,
    target_update_freq=1000,
    seed=0,
    update_ratio=5,
    network_size=(128,128),
    num_envs=24,
    max_episode_reward=500,
    return_steps=False
):
    """
    Trains a DQN on CartPole-v1 using various configurations:
      - naive: no replay buffer, no target network
      - only_tn: uses a target network, but no replay buffer
      - only_er: uses replay buffer, but no target network
      - tn_er: uses both replay buffer and target network
    Returns:
      episode_rewards (list of floats): returns from completed episodes.
      If return_steps is True, also returns (learning_curve, step_record) where:
         learning_curve is the list of episode returns,
         step_record is the corresponding environment steps at episode termination.
    """
    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create vectorized environments
    env = gym.vector.SyncVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(num_envs)])
    states, _ = env.reset(seed=seed)

    state_dim = env.single_observation_space.shape[0]
    action_dim = env.single_action_space.n
    
    # Create policy net
    policy_net = QNetwork(state_dim, action_dim, hidden_sizes=network_size).to(device)
    
    # Create target net if needed
    if config == "naive":
        target_net = None
    else:
        target_net = QNetwork(state_dim, action_dim, hidden_sizes=network_size).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    
    # Create replay buffer if needed
    if config in ["naive", "only_tn"]:
        replay_buffer = None
    else:
        replay_buffer = ReplayBuffer(buffer_capacity)

    episode_returns = np.zeros(num_envs, dtype=np.float32)
    episode_rewards = []  # Total return for each completed episode
    step_record = [] if return_steps else None

    epsilon = epsilon_start
    total_steps_counter = 0

    while total_steps_counter < total_timesteps:
        # Epsilon-greedy action selection
        state_tensor = torch.FloatTensor(states).to(device)
        with torch.no_grad():
            q_values = policy_net(state_tensor)
            best_actions = torch.argmax(q_values, dim=1).cpu().numpy()
        
        actions = []
        for i in range(num_envs):
            if np.random.rand() < epsilon:
                a = env.single_action_space.sample()
            else:
                a = int(best_actions[i])
            actions.append(a)

        next_states, rewards, dones, truncated, infos = env.step(actions)
        # Terminate an episode if the return exceeds max_episode_reward
        dones = np.logical_or(dones, episode_returns >= max_episode_reward)
        
        for i in range(num_envs):
            episode_returns[i] += rewards[i]
            
            if replay_buffer is not None:
                replay_buffer.push(states[i], actions[i], rewards[i], next_states[i], dones[i])
            else:
                transition = (states[i], actions[i], rewards[i], next_states[i], dones[i])
                update_single(policy_net, target_net, optimizer, gamma, transition)
            
            if dones[i]:
                episode_rewards.append(episode_returns[i])
                if return_steps:
                    step_record.append(total_steps_counter)
                episode_returns[i] = 0.0
        
        states = next_states
        total_steps_counter += num_envs

        if replay_buffer is not None and len(replay_buffer) >= batch_size:
            for _ in range(update_ratio):
                batch = replay_buffer.sample(batch_size)
                update_batch(policy_net, target_net, optimizer, gamma, batch)

        if target_net is not None and (total_steps_counter % target_update_freq == 0):
            target_net.load_state_dict(policy_net.state_dict())

        epsilon = max(epsilon_end, epsilon_start * np.exp(-total_steps_counter / epsilon_decay))

    env.close()
    if return_steps:
        return episode_rewards, episode_rewards, step_record
    else:
        return episode_rewards

###############################################################################
# Grid Search Function (uses 5 seeds for each setting)
###############################################################################
def run_grid_search():
    """
    Runs grid search over multiple hyperparameters and configurations.
    """
    configs = ["naive", "only_tn", "only_er", "tn_er"]
    learning_rates = [1e-4, 5e-4, 1e-3]
    epsilon_decays = [5000, 10000, 15000]
    batch_sizes = [32, 64, 128]
    results = {}

    for config in configs:
        for lr in learning_rates:
            for eps_decay in epsilon_decays:
                for bs in batch_sizes:
                    key = f"{config}_lr={lr}_eps={eps_decay}_bs={bs}"
                    logger.info(f"Running Grid Search: {key}")
                    all_final_scores = []
                    for seed in range(5):
                        rewards = train_cartpole_dqn(
                            config=config,
                            lr=lr,
                            epsilon_decay=eps_decay,
                            batch_size=bs,
                            seed=seed,
                            total_timesteps=100000
                        )
                        if len(rewards) >= 100:
                            final_score = np.mean(rewards[-100:])
                        else:
                            final_score = np.nan
                        all_final_scores.append(final_score)
                    overall_mean = np.nanmean(all_final_scores)
                    results[key] = overall_mean
                    logger.info(f"{key} - Mean Final Score: {overall_mean}")
    return results

###############################################################################
# Smoothing Function (Exponential Moving Average)
###############################################################################
def smooth(values, alpha=0.3):
    """
    Applies an exponential moving average (EMA) to smooth the input values.
    """
    smoothed_values = []
    if len(values) == 0:
        return values  # Return empty if no data
    smoothed_values.append(values[0])
    for i in range(1, len(values)):
        smoothed_values.append(alpha * values[i] + (1 - alpha) * smoothed_values[-1])
    return smoothed_values


###############################################################################
# Grid Search (Full Learning Curves) Function
##############################################################################

def run_grid_search_curves():
    """
    Runs grid search over selected hyperparameters for the 'tn_er' configuration.
    For each setting, runs 5 seeds and returns a dictionary:
         {hyperparam_string: averaged_learning_curve}
    The learning curve is a list of averaged episode returns across seeds.
    """
    learning_rates = [1e-4, 5e-4, 1e-3]
    epsilon_decays = [5000, 10000, 15000]
    batch_sizes = [32, 64, 128]
    results = {}
    
    for lr in learning_rates:
        for eps_decay in epsilon_decays:
            for bs in batch_sizes:
                key = f"lr={lr}_eps={eps_decay}_bs={bs}"
                logger.info(f"Running Grid Search Curves: {key}")
                all_rewards = []
                for seed in range(5):
                    rewards = train_cartpole_dqn(
                        config="tn_er",
                        lr=lr,
                        epsilon_decay=eps_decay,
                        batch_size=bs,
                        seed=seed,
                        total_timesteps=100000
                    )
                    all_rewards.append(rewards)
                # Align runs by truncating to the minimum length
                min_len = min(len(r) for r in all_rewards)
                aligned = np.array([r[:min_len] for r in all_rewards])
                mean_curve = np.mean(aligned, axis=0)
                results[key] = mean_curve.tolist()
                logger.info(f"{key} - Final Mean: {np.mean(mean_curve[-100:])}")
    return results
###############################################################################
# Smoothing Function (Exponential Moving Average)
###############################################################################
def smooth(values, alpha=0.3):
    """
    Applies an exponential moving average (EMA) to smooth the input list of values.
    """
    if not values or len(values) == 0:
        return values
    # Convert numpy array to list if necessary
    if isinstance(values, np.ndarray):
        values = values.tolist()
    smoothed = [values[0]]
    for v in values[1:]:
        smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])
    return smoothed
###############################################################################
# Plotting: Grid Search Learning Curves
###############################################################################
def plot_learning_curves(all_curves, smoothing_alpha=0.3):
    """
    Plots smoothed learning curves for each hyperparameter setting.
    :param all_curves: Dictionary {config_string: learning_curve (list of episode returns)}
    :param smoothing_alpha: Smoothing factor for the EMA.
    """
    plt.figure(figsize=(12, 8))
    
    for key, curve in all_curves.items():
        smoothed_curve = smooth(curve, alpha=smoothing_alpha)
        episodes = np.arange(len(smoothed_curve))
        plt.plot(episodes, smoothed_curve, label=key)
    
    plt.xlabel("Episode")
    plt.ylabel("Mean Return")
    plt.title("Grid Search: Smoothed Learning Curves")
    plt.legend(loc='best', fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/grid_search_learning_curves.png")
    plt.show()
###############################################################################
# 1) Configuration Comparison (runs 5 seeds per config)
###############################################################################
def run_config_comparison():
    """
    Runs each of the four configurations 5 times (using seeds 0-4) and returns a dictionary:
      config -> list of runs (each run is a list of episode rewards).
    """
    configs = ["naive", "only_tn", "only_er", "tn_er"]
    seeds = [0, 1, 2, 3, 4]
    all_results = {}
    for cfg in configs:
        logger.info(f"Running config comparison for: {cfg}")
        runs = []
        for seed in seeds:
            rewards = train_cartpole_dqn(
                config=cfg,
                total_timesteps=100000,
                seed=seed
            )
            runs.append(rewards)
        all_results[cfg] = runs
    return all_results

def plot_config_comparison(all_results):
    """
    Averages across the 5 seeds and plots the mean learning curve with standard deviation shading.
    """
    plt.figure(figsize=(10, 6))
    for cfg, runs in all_results.items():
        min_len = min(len(r) for r in runs)
        truncated_runs = np.array([r[:min_len] for r in runs])
        mean_rewards = np.mean(truncated_runs, axis=0)
        std_rewards = np.std(truncated_runs, axis=0)
        episodes = np.arange(1, min_len + 1)
        plt.plot(episodes, mean_rewards, label=cfg)
        plt.fill_between(episodes, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2)
    
    plt.xlabel("Episode")
    plt.ylabel("Episode Return")
    seeds = [0, 1, 2, 3, 4]
    ablation_results = {}
    for r in ratios:
        logger.info(f"Running ablation study for update_ratio = {r}")
        runs = []
        for seed in seeds:
            rewards = train_cartpole_dqn(
                config="tn_er",
                update_ratio=r,
                total_timesteps=100000, 
                seed=seed
            )
            runs.append(rewards)
        ablation_results[r] = runs
    return ablation_results

def plot_update_ratio_ablation(ablation_results):
    """
    Averages across the 5 seeds for each update_ratio value and plots the mean learning curve with std shading.
    """
    plt.figure(figsize=(10, 6))
    for ratio, runs in ablation_results.items():
        min_len = min(len(r) for r in runs)
        truncated_runs = np.array([r[:min_len] for r in runs])
        mean_rewards = np.mean(truncated_runs, axis=0)
        std_rewards = np.std(truncated_runs, axis=0)
        episodes = np.arange(1, min_len + 1)
        plt.plot(episodes, mean_rewards, label=f"Update Ratio = {ratio}")
        plt.fill_between(episodes, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2)
    
    plt.xlabel("Episode")
    plt.ylabel("Episode Return")
    plt.title("Ablation Study on Update Ratio (Averaged over 5 seeds)")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/update_ratio_ablation_avg.png")
    plt.show()

###############################################################################
# New Function for Q-Learning Learning Curve Plotting
###############################################################################
def run_q_learning_plot():
    """
    Runs Q-learning (using the 'naive' configuration) on CartPole-v1 over 5 seeds,
    and plots the aggregated (averaged and smoothed) learning curve (Return vs. Environment Steps).
    """
    config = "naive"
    total_timesteps = 100000
    seeds = [0, 1, 2, 3, 4]
    num_envs = 24
    
    all_learning_curves = []
    all_step_records = []
    
    for seed in seeds:
        # When return_steps=True, train_cartpole_dqn returns (episode_rewards, learning_curve, step_record)
        episode_rewards, learning_curve, step_record = train_cartpole_dqn(
            config=config,
            total_timesteps=total_timesteps,
            max_episode_reward=500,
            batch_size=256,
            gamma=0.99,
            lr=5e-4,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=10000,
            buffer_capacity=10000,
            target_update_freq=1000,
            seed=seed,
            update_ratio=5,
            network_size=(128,128),
            num_envs=num_envs,
            return_steps=True
        )
        all_learning_curves.append(learning_curve)
        all_step_records.append(step_record)
    
    # Align results by truncating to the minimum length
    min_length = min(len(lc) for lc in all_learning_curves)
    trimmed_learning_curves = [np.array(lc[:min_length]) for lc in all_learning_curves]
    trimmed_step_records = [np.array(sr[:min_length]) for sr in all_step_records]
    
    avg_learning_curve = np.mean(trimmed_learning_curves, axis=0)
    avg_step_record = np.mean(trimmed_step_records, axis=0)
    
    # Smooth the averaged learning curve
    smoothed_curve = smooth(avg_learning_curve, alpha=0.3)
    # Adjust steps to match the length of the smoothed curve
    smoothed_steps = avg_step_record[-len(smoothed_curve):]
    
    plt.plot(smoothed_steps, smoothed_curve, label="Q-Learning Return (5 seeds averaged)")

    # Force x-axis from 0 to 100,000
    plt.xlim(0, 100000)

    plt.xlabel("Environment Steps")
    plt.ylabel("Return (Smoothed)")
    plt.title("Q-Learning: Learning Curve on CartPole-v1 (up to 100,000 steps)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    logger.info("Saved Q-Learning learning curve plot: plots/q_learning_curve_5seeds.png")

###############################################################################
# Main Entry Point
###############################################################################
if __name__ == "__main__":
    # Uncomment the following lines to run other studies:
    config_results = run_config_comparison()
    plot_config_comparison(config_results)
    ablation_results = run_update_ratio_ablation(ratios=[1, 5, 10])
    plot_update_ratio_ablation(ablation_results)
    grid_search_results = run_grid_search()
    logger.info("Grid Search Results:")
    for k, v in grid_search_results.items():
        logger.info(f"{k}: {v}")
    plot_learning_curves(grid_search_results)
    
    # Run the Q-Learning learning curve plot
    #run_q_learning_plot()
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import csv
import time

# Create directory for saving CSVs and plots
os.makedirs("results", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Smoothing function: moving average with window 50
def smooth_curve(data, window=50):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode="valid")

# Define the Policy Network (outputs action probabilities)
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=(128, 128)):
        super(PolicyNetwork, self).__init__()
        layers = []
        input_dim = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, action_dim))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        logits = self.model(x)
        probs = torch.softmax(logits, dim=-1)
        return probs

# Define the Value Network (for AC and A2C)
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_sizes=(128, 128)):
        super(ValueNetwork, self).__init__()
        layers = []
        input_dim = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# Compute discounted returns for an episode
def compute_returns(rewards, gamma):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns

# Training function for REINFORCE
def train_reinforce(total_timesteps=1000000, gamma=0.99, lr=1e-3, network_size=(128,128), seed=0):
    env = gym.make("CartPole-v1")
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = PolicyNetwork(state_dim, action_dim, hidden_sizes=network_size).to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    
    episode_rewards = []
    episode_steps = []
    total_steps = 0
    episode = 0

    while total_steps < total_timesteps:
        state, _ = env.reset(seed=seed)
        done = False
        log_probs = []
        rewards = []
        ep_steps = 0
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            probs = policy_net(state_tensor)
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            log_prob = m.log_prob(action)
            next_state, reward, done, truncated, _ = env.step(action.item())
            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state
            ep_steps += 1
            total_steps += 1
            if done or truncated:
                break

        episode_reward = sum(rewards)
        episode_rewards.append(episode_reward)
        episode_steps.append(total_steps)

        returns = compute_returns(rewards, gamma)
        returns = torch.FloatTensor(returns).to(device)
        # Normalize returns for stability
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        loss = 0
        for log_prob, G in zip(log_probs, returns):
            loss += -log_prob * G
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        episode += 1
        if episode % 10 == 0:
            print(f"REINFORCE - Episode {episode}, Reward: {episode_reward}, Total Steps: {total_steps}")
    
    env.close()
    return episode_rewards, episode_steps

# Training function for Basic Actor-Critic (AC)
def train_actor_critic(total_timesteps=1000000, gamma=0.99, lr=1e-3, update_ratio=1, network_size=(128,128), seed=42):
    env = gym.make("CartPole-v1")
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = PolicyNetwork(state_dim, action_dim, hidden_sizes=network_size).to(device)
    value_net = ValueNetwork(state_dim, hidden_sizes=network_size).to(device)
    
    optimizer_policy = optim.Adam(policy_net.parameters(), lr=lr)
    optimizer_value = optim.Adam(value_net.parameters(), lr=lr)
    
    episode_rewards = []
    episode_steps = []
    total_steps = 0
    episode = 0

    while total_steps < total_timesteps:
        state, _ = env.reset(seed=seed+episode)
        done = False
        log_probs = []
        rewards = []
        values = []
        ep_steps = 0
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            probs = policy_net(state_tensor)
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            log_prob = m.log_prob(action)
            value = value_net(state_tensor)
            
            next_state, reward, done, truncated, _ = env.step(action.item())
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            state = next_state
            ep_steps += 1
            total_steps += 1
            if done or truncated:
                break

        episode_reward = sum(rewards)
        episode_rewards.append(episode_reward)
        episode_steps.append(total_steps)
        
        returns = compute_returns(rewards, gamma)
        returns = torch.FloatTensor(returns).to(device)
        values = torch.cat(values).squeeze()
        advantages = returns - values
        
        policy_loss = (-torch.stack(log_probs) * advantages.detach()).sum()
        value_loss = advantages.pow(2).sum()
        
        # Combine losses and perform multiple updates
        for i in range(update_ratio):
            optimizer_policy.zero_grad()
            optimizer_value.zero_grad()
            total_loss = policy_loss + value_loss
            if i < update_ratio - 1:
                total_loss.clone().backward(retain_graph=True)
            else:
                total_loss.clone().backward()
            optimizer_policy.step()
            optimizer_value.step()
        
        episode += 1
        if episode % 10 == 0:
            print(f"Actor-Critic - Episode {episode}, Reward: {episode_reward}, Total Steps: {total_steps}")
    
    env.close()
    return episode_rewards, episode_steps

# Training function for Advantage Actor-Critic (A2C)
def train_a2c(total_timesteps=1000000, gamma=0.99, lr=1e-3, update_ratio=1, network_size=(128,128), seed=100):
    env = gym.make("CartPole-v1")
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    policy_net = PolicyNetwork(state_dim, action_dim, hidden_sizes=network_size).to(device)
    value_net = ValueNetwork(state_dim, hidden_sizes=network_size).to(device)
    
    optimizer_policy = optim.Adam(policy_net.parameters(), lr=lr)
    optimizer_value = optim.Adam(value_net.parameters(), lr=lr)
    
    episode_rewards = []
    episode_steps = []
    total_steps = 0
    episode = 0

    while total_steps < total_timesteps:
        state, _ = env.reset(seed=seed+episode)
        done = False
        log_probs = []
        rewards = []
        values = []
        ep_steps = 0
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            probs = policy_net(state_tensor)
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            log_prob = m.log_prob(action)
            value = value_net(state_tensor)
            
            next_state, reward, done, truncated, _ = env.step(action.item())
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            state = next_state
            ep_steps += 1
            total_steps += 1
            if done or truncated:
                break

        episode_reward = sum(rewards)
        episode_rewards.append(episode_reward)
        episode_steps.append(total_steps)
        
        returns = compute_returns(rewards, gamma)
        returns = torch.FloatTensor(returns).to(device)
        values = torch.cat(values).squeeze()
        advantages = returns - values
        
        policy_loss = (-torch.stack(log_probs) * advantages.detach()).sum()
        value_loss = advantages.pow(2).sum()
        
        # Combine losses and perform multiple updates
        for i in range(update_ratio):
            optimizer_policy.zero_grad()
            optimizer_value.zero_grad()
            total_loss = policy_loss + value_loss
            if i < update_ratio - 1:
                total_loss.clone().backward(retain_graph=True)
            else:
                total_loss.clone().backward()
            optimizer_policy.step()
            optimizer_value.step()
        
        episode += 1
        if episode % 10 == 0:
            print(f"A2C - Episode {episode}, Reward: {episode_reward}, Total Steps: {total_steps}")
    
    env.close()
    return episode_rewards, episode_steps

# Debug function to check if the data is valid
def debug_results_data(results_dict):
    print("\n=== DEBUGGING RESULTS DATA ===")
    for label, (rewards, steps) in results_dict.items():
        print(f"Algorithm: {label}")
        print(f"  Rewards length: {len(rewards)}")
        print(f"  Rewards min: {min(rewards)}, max: {max(rewards)}, mean: {np.mean(rewards)}")
        print(f"  Steps length: {len(steps)}")
        print(f"  First 5 rewards: {rewards[:5]}")
        print(f"  Last 5 rewards: {rewards[-5:]}")
        
    # Check if any algorithms have identical data
    algorithms = list(results_dict.keys())
    for i in range(len(algorithms)):
        for j in range(i+1, len(algorithms)):
            alg1 = algorithms[i]
            alg2 = algorithms[j]
            rewards1 = results_dict[alg1][0]
            rewards2 = results_dict[alg2][0]
            
            if len(rewards1) == len(rewards2):
                if len(rewards1) > 0:
                    similarity = np.mean(np.abs(np.array(rewards1) - np.array(rewards2)))
                    print(f"Similarity between {alg1} and {alg2}: {similarity}")
                    if similarity < 1e-5:
                        print(f"WARNING: {alg1} and {alg2} have nearly identical data!")

# Enhanced plotting function with more debugging and improved visibility
def plot_learning_curves(results_dict, smoothing_window=50):
    plt.figure(figsize=(12,8))
    
    
    styles = {
    "REINFORCE": {"color": "blue", "linestyle": "-", "linewidth": 2, "marker": None},
    "Actor-Critic": {"color": "red", "linestyle": "-", "linewidth": 3, "marker": "o", "markevery": 50},
    "A2C": {"color": "green", "linestyle": "-", "linewidth": 2, "marker": None}
}

    
    print("\n=== PLOTTING INFORMATION ===")
    print(f"Algorithms to plot: {list(results_dict.keys())}")
    
    for label, (rewards, steps) in results_dict.items():
        print(f"Processing {label} with {len(rewards)} rewards")
        
        smoothed = smooth_curve(rewards, window=smoothing_window)
        episodes = np.arange(len(smoothed))
        
        print(f"  After smoothing: {len(smoothed)} points")
        print(f"  Smoothed data range: min={np.min(smoothed)}, max={np.max(smoothed)}")
        
        style = styles.get(label, {"color": "black", "linestyle": "-", "linewidth": 1, "marker": None})
        
        plt.plot(episodes, smoothed, 
                 label=f"{label} (min={np.min(smoothed):.1f}, max={np.max(smoothed):.1f})", 
                 color=style["color"], 
                 linestyle=style["linestyle"], 
                 linewidth=style["linewidth"],
                 marker=style["marker"],
                 markevery=style.get("markevery", None))
    
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Smoothed Total Reward (window=50)", fontsize=12)
    plt.title("Learning Curves", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig("plots/learning_curves.png", dpi=300)
    plt.show()

# Function to plot final performance bar chart (average over last 100 episodes)
def plot_final_performance(results_dict):
    labels = []
    final_perfs = []
    for label, (rewards, _) in results_dict.items():
        if len(rewards) >= 100:
            avg_final = np.mean(rewards[-100:])
        else:
            avg_final = np.mean(rewards)
        labels.append(label)
        final_perfs.append(avg_final)
    plt.figure(figsize=(8,6))
    plt.bar(labels, final_perfs)
    plt.xlabel("Algorithm")
    plt.ylabel("Average Reward (Last 100 Episodes)")
    plt.title("Final Performance Comparison")
    plt.tight_layout()
    plt.savefig("plots/final_performance.png")
    plt.show()

# Function to save results to CSV file
def save_results_csv(filename, episodes, rewards, steps):
    df = pd.DataFrame({
        "Episode": episodes,
        "Reward": rewards,
        "Total_Steps": steps
    })
    df.to_csv(filename, index=False)
    print(f"Saved CSV: {filename}")

# Main entry point
if __name__ == "__main__":
    overall_start_time = time.time()
    
    results = {}
    
    # --- Run REINFORCE ---
    print("Training with REINFORCE...")
    rewards_reinforce, steps_reinforce = train_reinforce(total_timesteps=1000000, gamma=0.99, lr=1e-3, network_size=(128,128), seed=0)
    results["REINFORCE"] = (rewards_reinforce, steps_reinforce)
    save_results_csv("results/reinforce_results.csv", np.arange(1, len(rewards_reinforce)+1), rewards_reinforce, steps_reinforce)
    
    # --- Run Actor-Critic ---
    print("Training with Actor-Critic...")
    rewards_ac, steps_ac = train_actor_critic(total_timesteps=1000000, gamma=0.99, lr=1e-3, update_ratio=1, network_size=(128,128), seed=42)
    results["Actor-Critic"] = (rewards_ac, steps_ac)
    save_results_csv("results/actor_critic_results.csv", np.arange(1, len(rewards_ac)+1), rewards_ac, steps_ac)
    
    # --- Run A2C ---
    print("Training with A2C...")
    rewards_a2c, steps_a2c = train_a2c(total_timesteps=1000000, gamma=0.99, lr=1e-3, update_ratio=1, network_size=(128,128), seed=100)
    results["A2C"] = (rewards_a2c, steps_a2c)
    save_results_csv("results/a2c_results.csv", np.arange(1, len(rewards_a2c)+1), rewards_a2c, steps_a2c)
    
    # Debug the results data to check for issues
    debug_results_data(results)
    
    print("Algorithms in results dictionary:", list(results.keys()))
    
    # Plot learning curves and final performance
    plot_learning_curves(results, smoothing_window=50)
    plot_final_performance(results)
    
    overall_end_time = time.time()
    print(f"Overall training time: {overall_end_time - overall_start_time:.2f} seconds")

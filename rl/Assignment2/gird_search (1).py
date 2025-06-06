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
import traceback

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
            layers.append(nn.ReLU())  # default non-inplace
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
            layers.append(nn.ReLU())  # default non-inplace
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
def train_reinforce(total_timesteps=100000, gamma=0.99, lr=1e-3, network_size=(128,128), seed=0):
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
        state, _ = env.reset(seed=seed+episode)  # Use different seed for each episode
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
def train_actor_critic(total_timesteps=100000, gamma=0.99, lr=1e-3, update_ratio=1, network_size=(128,128), seed=0):
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
        state, _ = env.reset(seed=seed+episode)  # Use different seed for each episode
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
def train_a2c(total_timesteps=100000, gamma=0.99, lr=1e-3, update_ratio=1, network_size=(128,128), seed=0):
    env = gym.make("CartPole-v1")
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n  # Corrected
    
    policy_net = PolicyNetwork(state_dim, action_dim, hidden_sizes=network_size).to(device)
    value_net = ValueNetwork(state_dim, hidden_sizes=network_size).to(device)
    
    optimizer_policy = optim.Adam(policy_net.parameters(), lr=lr)
    optimizer_value = optim.Adam(value_net.parameters(), lr=lr)
    
    episode_rewards = []
    episode_steps = []
    total_steps = 0
    episode = 0

    while total_steps < total_timesteps:
        state, _ = env.reset(seed=seed+episode)  # Use different seed for each episode
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
        advantages = returns - values  # Advantage explicitly
        
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

# Function to save results to CSV file
def save_results_csv(filename, episodes, rewards, steps):
    df = pd.DataFrame({
        "Episode": episodes,
        "Reward": rewards,
        "Total_Steps": steps
    })
    df.to_csv(filename, index=False)
    print(f"Saved CSV: {filename}")

# ------------------------------
# Grid search functions for each algorithm

def grid_search_a2c(total_timesteps=100000, gamma=0.99, reduced_search=False):
    # Use reduced parameter set for faster testing if requested
    if reduced_search:
        learning_rates = [1e-3]
        update_ratios = [1, 5]
        network_sizes = [(128,128)]
        seeds = [0, 1]
    else:
        learning_rates = [1e-4, 5e-4, 1e-3]
        update_ratios = [1, 5, 10]
        network_sizes = [(64,64), (128,128), (256,256)]
        seeds = [0, 1, 2, 3, 4]
    
    grid_results = {}
    grid_learning_curves = {}
    
    try:
        for lr in learning_rates:
            for ur in update_ratios:
                for net_size in network_sizes:
                    key = f"A2C_lr={lr}_ur={ur}_net={net_size[0]}x{net_size[1]}"
                    all_rewards = []
                    all_steps = []
                    print(f"Grid Search (A2C) - Combination: {key}")
                    
                    try:
                        for seed in seeds:
                            rewards, steps = train_a2c(total_timesteps=total_timesteps, gamma=gamma, 
                                                      lr=lr, update_ratio=ur, network_size=net_size, seed=seed)
                            all_rewards.append(np.array(rewards))
                            all_steps.append(np.array(steps))
                            
                        # Save intermediate results after each parameter combination
                        min_len = min(len(r) for r in all_rewards)
                        trimmed_rewards = np.stack([r[:min_len] for r in all_rewards])
                        avg_rewards = np.mean(trimmed_rewards, axis=0)
                        final_perf = np.mean(avg_rewards[-100:]) if len(avg_rewards) >= 100 else np.mean(avg_rewards)
                        grid_results[key] = final_perf
                        grid_learning_curves[key] = avg_rewards
                        
                        # Save intermediate results to CSV
                        intermediate_df = pd.DataFrame({"Combination": [key], "Final_Performance": [final_perf]})
                        intermediate_df.to_csv(f"results/grid_search_a2c_intermediate_{key}.csv", index=False)
                        print(f"Saved intermediate result for {key}: {final_perf}")
                        
                    except Exception as e:
                        print(f"Error in A2C grid search for {key}: {str(e)}")
                        traceback.print_exc()
                        continue
    
        # Plot learning curves
        plt.figure(figsize=(12,8))
        for key, curve in grid_learning_curves.items():
            smoothed_curve_vals = smooth_curve(curve, window=50)
            episodes = np.arange(len(smoothed_curve_vals))
            plt.plot(episodes, smoothed_curve_vals, label=key)
        plt.xlabel("Episode")
        plt.ylabel("Smoothed Reward (window=50)")
        plt.title("Grid Search Learning Curves (A2C)")
        plt.legend(fontsize=8, loc="lower right")
        plt.tight_layout()
        plt.savefig("plots/grid_search_a2c_learning_curves.png")
        plt.show()
        
        # Plot final performance
        plt.figure(figsize=(14,6))
        keys = list(grid_results.keys())
        final_vals = [grid_results[k] for k in keys]
        plt.bar(keys, final_vals)
        plt.xlabel("Hyperparameter Combination")
        plt.ylabel("Final Performance (Avg Reward of Last 100 Episodes)")
        plt.title("Grid Search Final Performance (A2C)")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig("plots/grid_search_a2c_final_performance.png")
        plt.show()
        
        # Save final results
        df_grid = pd.DataFrame({"Combination": list(grid_results.keys()), "Final_Performance": list(grid_results.values())})
        df_grid.to_csv("results/grid_search_a2c_results.csv", index=False)
        print("Saved grid search CSV: results/grid_search_a2c_results.csv")
        
    except Exception as e:
        print(f"Error in A2C grid search: {str(e)}")
        traceback.print_exc()
    
    return grid_results, grid_learning_curves

def grid_search_reinforce(total_timesteps=100000, gamma=0.99, reduced_search=False):
    # Use reduced parameter set for faster testing if requested
    if reduced_search:
        learning_rates = [1e-3]
        network_sizes = [(128,128)]
        seeds = [0, 1]
    else:
        learning_rates = [1e-4, 5e-4, 1e-3]
        network_sizes = [(64,64), (128,128), (256,256)]
        seeds = [0, 1, 2, 3, 4]
    
    grid_results = {}
    grid_learning_curves = {}
    
    try:
        for lr in learning_rates:
            for net_size in network_sizes:
                key = f"REINFORCE_lr={lr}_net={net_size[0]}x{net_size[1]}"
                all_rewards = []
                all_steps = []
                print(f"Grid Search (REINFORCE) - Combination: {key}")
                
                try:
                    for seed in seeds:
                        rewards, steps = train_reinforce(total_timesteps=total_timesteps, gamma=gamma, 
                                                        lr=lr, network_size=net_size, seed=seed)
                        all_rewards.append(np.array(rewards))
                        all_steps.append(np.array(steps))
                    
                    # Save intermediate results after each parameter combination
                    min_len = min(len(r) for r in all_rewards)
                    trimmed_rewards = np.stack([r[:min_len] for r in all_rewards])
                    avg_rewards = np.mean(trimmed_rewards, axis=0)
                    final_perf = np.mean(avg_rewards[-100:]) if len(avg_rewards) >= 100 else np.mean(avg_rewards)
                    grid_results[key] = final_perf
                    grid_learning_curves[key] = avg_rewards
                    
                    # Save intermediate results to CSV
                    intermediate_df = pd.DataFrame({"Combination": [key], "Final_Performance": [final_perf]})
                    intermediate_df.to_csv(f"results/grid_search_reinforce_intermediate_{key}.csv", index=False)
                    print(f"Saved intermediate result for {key}: {final_perf}")
                    
                except Exception as e:
                    print(f"Error in REINFORCE grid search for {key}: {str(e)}")
                    traceback.print_exc()
                    continue
        
        # Plot learning curves
        plt.figure(figsize=(12,8))
        for key, curve in grid_learning_curves.items():
            smoothed_curve_vals = smooth_curve(curve, window=50)
            episodes = np.arange(len(smoothed_curve_vals))
            plt.plot(episodes, smoothed_curve_vals, label=key)
        plt.xlabel("Episode")
        plt.ylabel("Smoothed Reward (window=50)")
        plt.title("Grid Search Learning Curves (REINFORCE)")
        plt.legend(fontsize=8, loc="lower right")
        plt.tight_layout()
        plt.savefig("plots/grid_search_reinforce_learning_curves.png")
        plt.show()
        
        # Plot final performance
        plt.figure(figsize=(14,6))
        keys = list(grid_results.keys())
        final_vals = [grid_results[k] for k in keys]
        plt.bar(keys, final_vals)
        plt.xlabel("Hyperparameter Combination")
        plt.ylabel("Final Performance (Avg Reward of Last 100 Episodes)")
        plt.title("Grid Search Final Performance (REINFORCE)")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig("plots/grid_search_reinforce_final_performance.png")
        plt.show()
        
        # Save final results
        df_grid = pd.DataFrame({"Combination": list(grid_results.keys()), "Final_Performance": list(grid_results.values())})
        df_grid.to_csv("results/grid_search_reinforce_results.csv", index=False)
        print("Saved grid search CSV: results/grid_search_reinforce_results.csv")
        
    except Exception as e:
        print(f"Error in REINFORCE grid search: {str(e)}")
        traceback.print_exc()
    
    return grid_results, grid_learning_curves

def grid_search_actor_critic(total_timesteps=100000, gamma=0.99, reduced_search=False):
    # Use reduced parameter set for faster testing if requested
    if reduced_search:
        learning_rates = [1e-3]
        update_ratios = [1, 5]
        network_sizes = [(128,128)]
        seeds = [0, 1]
    else:
        learning_rates = [1e-4, 5e-4, 1e-3]
        update_ratios = [1, 5, 10]
        network_sizes = [(64,64), (128,128), (256,256)]
        seeds = [0, 1, 2, 3, 4]
    
    grid_results = {}
    grid_learning_curves = {}
    
    try:
        for lr in learning_rates:
            for ur in update_ratios:
                for net_size in network_sizes:
                    key = f"AC_lr={lr}_ur={ur}_net={net_size[0]}x{net_size[1]}"
                    all_rewards = []
                    all_steps = []
                    print(f"Grid Search (Actor-Critic) - Combination: {key}")
                    
                    try:
                        for seed in seeds:
                            rewards, steps = train_actor_critic(total_timesteps=total_timesteps, gamma=gamma, 
                                                              lr=lr, update_ratio=ur, network_size=net_size, seed=seed)
                            all_rewards.append(np.array(rewards))
                            all_steps.append(np.array(steps))
                        
                        # Save intermediate results after each parameter combination
                        min_len = min(len(r) for r in all_rewards)
                        trimmed_rewards = np.stack([r[:min_len] for r in all_rewards])
                        avg_rewards = np.mean(trimmed_rewards, axis=0)
                        final_perf = np.mean(avg_rewards[-100:]) if len(avg_rewards) >= 100 else np.mean(avg_rewards)
                        grid_results[key] = final_perf
                        grid_learning_curves[key] = avg_rewards
                        
                        # Save intermediate results to CSV
                        intermediate_df = pd.DataFrame({"Combination": [key], "Final_Performance": [final_perf]})
                        intermediate_df.to_csv(f"results/grid_search_actor_critic_intermediate_{key}.csv", index=False)
                        print(f"Saved intermediate result for {key}: {final_perf}")
                        
                    except Exception as e:
                        print(f"Error in Actor-Critic grid search for {key}: {str(e)}")
                        traceback.print_exc()
                        continue
        
        # Plot learning curves
        plt.figure(figsize=(12,8))
        for key, curve in grid_learning_curves.items():
            smoothed_curve_vals = smooth_curve(curve, window=50)
            episodes = np.arange(len(smoothed_curve_vals))
            plt.plot(episodes, smoothed_curve_vals, label=key)
        plt.xlabel("Episode")
        plt.ylabel("Smoothed Reward (window=50)")
        plt.title("Grid Search Learning Curves (Actor-Critic)")
        plt.legend(fontsize=8, loc="lower right")
        plt.tight_layout()
        plt.savefig("plots/grid_search_actor_critic_learning_curves.png")
        plt.show()
        
        # Plot final performance
        plt.figure(figsize=(14,6))
        keys = list(grid_results.keys())
        final_vals = [grid_results[k] for k in keys]
        plt.bar(keys, final_vals)
        plt.xlabel("Hyperparameter Combination")
        plt.ylabel("Final Performance (Avg Reward of Last 100 Episodes)")
        plt.title("Grid Search Final Performance (Actor-Critic)")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig("plots/grid_search_actor_critic_final_performance.png")
        plt.show()
        
        # Save final results
        df_grid = pd.DataFrame({"Combination": list(grid_results.keys()), "Final_Performance": list(grid_results.values())})
        df_grid.to_csv("results/grid_search_actor_critic_results.csv", index=False)
        print("Saved grid search CSV: results/grid_search_actor_critic_results.csv")
        
    except Exception as e:
        print(f"Error in Actor-Critic grid search: {str(e)}")
        traceback.print_exc()
    
    return grid_results, grid_learning_curves

# Combined grid search plotting for all three algorithms
def plot_all_grid_search_results(grid_curves_a2c, grid_curves_reinforce, grid_curves_ac, smoothing_window=50):
    try:
        fig, axs = plt.subplots(3,1, figsize=(12, 18))
        
        # Plot for A2C
        for key, curve in grid_curves_a2c.items():
            smoothed = smooth_curve(curve, window=smoothing_window)
            axs[0].plot(np.arange(len(smoothed)), smoothed, label=key)
        axs[0].set_title("Grid Search Learning Curves (A2C)")
        axs[0].set_xlabel("Episode")
        axs[0].set_ylabel("Smoothed Reward")
        axs[0].legend(fontsize=8, loc="lower right")
        axs[0].grid(True)
        
        # Plot for REINFORCE
        for key, curve in grid_curves_reinforce.items():
            smoothed = smooth_curve(curve, window=smoothing_window)
            axs[1].plot(np.arange(len(smoothed)), smoothed, label=key)
        axs[1].set_title("Grid Search Learning Curves (REINFORCE)")
        axs[1].set_xlabel("Episode")
        axs[1].set_ylabel("Smoothed Reward")
        axs[1].legend(fontsize=8, loc="lower right")
        axs[1].grid(True)
        
        # Plot for Actor-Critic
        for key, curve in grid_curves_ac.items():
            smoothed = smooth_curve(curve, window=smoothing_window)
            axs[2].plot(np.arange(len(smoothed)), smoothed, label=key)
        axs[2].set_title("Grid Search Learning Curves (Actor-Critic)")
        axs[2].set_xlabel("Episode")
        axs[2].set_ylabel("Smoothed Reward")
        axs[2].legend(fontsize=8, loc="lower right")
        axs[2].grid(True)
        
        plt.tight_layout()
        plt.savefig("plots/all_grid_search_learning_curves.png", dpi=300)
        plt.show()
        
    except Exception as e:
        print(f"Error in combined grid search plotting: {str(e)}")
        traceback.print_exc()

def plot_all_grid_final_performance(grid_results_a2c, grid_results_reinforce, grid_results_ac):
    try:
        labels = []
        final_vals = []
        colors = []
        
        # Add A2C results with green color
        for key, val in grid_results_a2c.items():
            labels.append(key)
            final_vals.append(val)
            colors.append('green')
            
        # Add REINFORCE results with blue color
        for key, val in grid_results_reinforce.items():
            labels.append(key)
            final_vals.append(val)
            colors.append('blue')
            
        # Add Actor-Critic results with red color
        for key, val in grid_results_ac.items():
            labels.append(key)
            final_vals.append(val)
            colors.append('red')
            
        plt.figure(figsize=(14,6))
        plt.bar(labels, final_vals, color=colors)
        plt.xlabel("Hyperparameter Combination")
        plt.ylabel("Final Performance (Avg Reward of Last 100 Episodes)")
        plt.title("Grid Search Final Performance (All Algorithms)")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig("plots/all_grid_search_final_performance.png", dpi=300)
        plt.show()
        
        # Save combined results to CSV
        combined_df = pd.DataFrame({
            "Combination": labels,
            "Final_Performance": final_vals,
            "Algorithm": [key.split('_')[0] for key in labels]
        })
        combined_df.to_csv("results/all_grid_search_results.csv", index=False)
        print("Saved combined grid search CSV: results/all_grid_search_results.csv")
        
    except Exception as e:
        print(f"Error in combined final performance plotting: {str(e)}")
        traceback.print_exc()

# Main entry point
if __name__ == "__main__":
    overall_start_time = time.time()
    
    # Ask user if they want to run a reduced grid search (faster)
    reduced_search = input("Run reduced grid search for faster results? (y/n): ").lower() == 'y'
    if reduced_search:
        print("Running reduced grid search with fewer hyperparameters and seeds...")
    else:
        print("Running full grid search (this will take a long time)...")
    
    try:
        # Run Grid Search for all algorithms
        print("\nRunning grid search for A2C...")
        grid_results_a2c, grid_curves_a2c = grid_search_a2c(total_timesteps=100000, gamma=0.99, reduced_search=reduced_search)
        
        print("\nRunning grid search for REINFORCE...")
        grid_results_reinforce, grid_curves_reinforce = grid_search_reinforce(total_timesteps=100000, gamma=0.99, reduced_search=reduced_search)
        
        print("\nRunning grid search for Actor-Critic...")
        grid_results_ac, grid_curves_ac = grid_search_actor_critic(total_timesteps=100000, gamma=0.99, reduced_search=reduced_search)
        
        # Plot combined grid search results for all algorithms
        print("\nPlotting combined grid search results...")
        plot_all_grid_search_results(grid_curves_a2c, grid_curves_reinforce, grid_curves_ac, smoothing_window=50)
        plot_all_grid_final_performance(grid_results_a2c, grid_results_reinforce, grid_results_ac)
        
        # Find and print best hyperparameters for each algorithm
        print("\n=== Best Hyperparameters ===")
        
        if grid_results_a2c:
            best_a2c = max(grid_results_a2c.items(), key=lambda x: x[1])
            print(f"Best A2C: {best_a2c[0]} with performance {best_a2c[1]:.2f}")
        
        if grid_results_reinforce:
            best_reinforce = max(grid_results_reinforce.items(), key=lambda x: x[1])
            print(f"Best REINFORCE: {best_reinforce[0]} with performance {best_reinforce[1]:.2f}")
        
        if grid_results_ac:
            best_ac = max(grid_results_ac.items(), key=lambda x: x[1])
            print(f"Best Actor-Critic: {best_ac[0]} with performance {best_ac[1]:.2f}")
        
        # Compare best algorithms
        best_algs = []
        if grid_results_a2c:
            best_algs.append(("A2C", max(grid_results_a2c.values())))
        if grid_results_reinforce:
            best_algs.append(("REINFORCE", max(grid_results_reinforce.values())))
        if grid_results_ac:
            best_algs.append(("Actor-Critic", max(grid_results_ac.values())))
        
        if best_algs:
            best_algorithm = max(best_algs, key=lambda x: x[1])
            print(f"\nOverall best algorithm: {best_algorithm[0]} with performance {best_algorithm[1]:.2f}")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        traceback.print_exc()
    
    overall_end_time = time.time()
    elapsed = overall_end_time - overall_start_time
    print(f"\nTotal execution time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")

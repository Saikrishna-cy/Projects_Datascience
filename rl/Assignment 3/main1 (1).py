import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import csv

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=(128,128)):
        super().__init__()
        layers = []
        in_dim = state_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        self.shared = nn.Sequential(*layers)
        self.policy_logits = nn.Linear(in_dim, action_dim)
        self.value = nn.Linear(in_dim, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.policy_logits(x), self.value(x)

def train_ppo(
    env_name="CartPole-v1",
    total_timesteps=100_000,
    batch_size=1024,
    mini_batch_size=64,
    epochs=20,
    gamma=0.99,
    lam=0.97,
    clip_eps=0.3,
    lr=5e-4,
    lr_decay=True,
    entropy_coef=0.02,
    seed=42,
    log_interval=100
):
    # Environment and seeds
    env = gym.make(env_name)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)
    env.action_space.seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Model and optimizer
    model = ActorCritic(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if lr_decay:
        updates = total_timesteps // batch_size
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lambda i: 1 - min(i, updates)/updates
        )
    else:
        scheduler = None

    # Logging containers
    episode_rewards = []
    episode_steps = []
    timestep_count = 0
    episode_count = 0
    obs = env.reset()
    curr_reward = 0.0

    # Main loop
    while timestep_count < total_timesteps:
        # Collect a batch of transitions
        states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []
        t = 0
        while t < batch_size and timestep_count < total_timesteps:
            s = torch.from_numpy(obs).float().unsqueeze(0)
            with torch.no_grad():
                logits, value = model(s)
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
                lp = dist.log_prob(a)

            nxt, r, done, _ = env.step(a.item())

            states.append(obs)
            actions.append(a.item())
            rewards.append(r)
            dones.append(done)
            log_probs.append(lp.item())
            values.append(value.item())

            curr_reward += r
            obs = nxt
            t += 1
            timestep_count += 1

            if done:
                episode_rewards.append(curr_reward)
                episode_steps.append(timestep_count)
                episode_count += 1
                if episode_count % log_interval == 0:
                    avg = np.mean(episode_rewards[-log_interval:])
                    print(f"Episode {episode_count}, Timesteps {timestep_count}/{total_timesteps}, "
                          f"Avg reward last {log_interval}: {avg:.2f}")
                curr_reward = 0.0
                obs = env.reset()

        # Compute value for last state if not done
        if dones[-1]:
            next_value = 0.0
        else:
            s = torch.from_numpy(obs).float().unsqueeze(0)
            with torch.no_grad():
                _, v = model(s)
                next_value = v.item()

        # GAE advantage and returns
        returns, advs = [], []
        gae = 0.0
        for i in reversed(range(len(rewards))):
            mask = 1.0 - float(dones[i])
            delta = rewards[i] + gamma * next_value * mask - values[i]
            gae = delta + gamma * lam * mask * gae
            advs.append(gae)
            returns.append(gae + values[i])
            next_value = values[i]
        advs.reverse(); returns.reverse()

        advs = np.array(advs, dtype=np.float32)
        returns = np.array(returns, dtype=np.float32)
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        # Convert to tensors
        S = torch.tensor(np.array(states), dtype=torch.float32)
        A = torch.tensor(actions, dtype=torch.int64)
        LP = torch.tensor(log_probs, dtype=torch.float32)
        R = torch.tensor(returns, dtype=torch.float32)
        Adv = torch.tensor(advs, dtype=torch.float32)

        # PPO update
        idxs = np.arange(len(states))
        for _ in range(epochs):
            np.random.shuffle(idxs)
            for start in range(0, len(states), mini_batch_size):
                mb = idxs[start:start+mini_batch_size]
                ms, ma = S[mb], A[mb]
                mlp, mr, madv = LP[mb], R[mb], Adv[mb]

                logits, vpred = model(ms)
                dist = torch.distributions.Categorical(logits=logits)
                new_lp = dist.log_prob(ma)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_lp - mlp)
                obj = torch.min(ratio * madv,
                               torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * madv)
                policy_loss = -obj.mean()

                vpred = vpred.squeeze(1)
                value_loss = ((mr - vpred) ** 2).mean()

                loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
        if scheduler:
            scheduler.step()

    env.close()

    # Write CSV
    with open("ppo_cartpole_episode_rewards.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Reward", "Total_Steps"])
        for i, (r, s) in enumerate(zip(episode_rewards, episode_steps), start=1):
            writer.writerow([i, r, s])

    # Plot learning curve
    window = 50
    cum = np.cumsum(np.insert(episode_rewards, 0, 0))
    if len(episode_rewards) >= window:
        smooth = (cum[window:] - cum[:-window]) / window
    else:
        smooth = episode_rewards

    plt.figure(figsize=(8,5), dpi=150)
    plt.plot(episode_steps, episode_rewards, label="Episode Reward", alpha=0.4)
    plt.plot(episode_steps[window-1:], smooth, label=f"{window}-Episode MA", linewidth=2)
    plt.xlabel("Total Steps", fontsize=12)
    plt.ylabel("Reward", fontsize=12)
    plt.title("PPO on CartPole-v1", fontsize=14)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("ppo_cartpole_learning_curve.png")
    plt.close()

    return episode_rewards, smooth.tolist()

if __name__ == "__main__":
    
    train_ppo()
    pass

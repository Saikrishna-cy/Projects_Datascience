# PPO for CartPole-v1 - Reinforcement Learning Assignment 3

This project implements Proximal Policy Optimization (PPO) from scratch using PyTorch to train an agent on the CartPole-v1 environment from OpenAI Gym. It includes stability enhancements like Generalized Advantage Estimation (GAE), entropy regularization, multi-epoch updates, and clipped policy ratios for safer gradient updates.


## Experiments

PPO Agent Training (Assignment 3)**  
- CartPole-v1 with discrete action space  
- Train for 100k timesteps with mini-batch PPO  
- Learning curve plotted and saved  
- Results logged to CSV  
- Final comparison with A2C, REINFORCE, and DQN baselines (external)  


## Project Structure

- `main1.py`: PPO training implementation and execution script  
- `requirements.txt`: Required Python packages with version constraints and descriptions  
- `ppo_cartpole_episode_rewards.csv`: Logs of episode rewards and timesteps  
- `ppo_cartpole_learning_curve.png`: Plot of reward progression  
- `README.md`: This documentation  

## Setup Instructions

1. Install Python 3.8 or higher  
2. (Optional) Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

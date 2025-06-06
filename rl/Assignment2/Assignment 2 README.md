Assignment 2: README

REINFORCE, A2C, and Actor-Critic on the CartPole-v1 Environment
This assignment implements three reinforcement learning algorithms on the CartPole-v1 environment:
1.	REINFORCE – A policy gradient algorithm that uses Monte Carlo estimates for the value function.
2.	Actor-Critic (AC) – Combines a policy network with a separate value network to stabilize learning.
3.	Advantage Actor-Critic (A2C) – Improves stability by using the advantage function to refine updates.
Experiments
1.	REINFORCE: Implements policy gradient updates using normalized Monte Carlo returns.
2.	Actor-Critic (AC): • Uses a distinct value network alongside the policy network. • Explains why the Q-learning loss is unsuitable, and presents an alternative loss function.
3.	Advantage Actor-Critic (A2C): • Uses the advantage function (difference between Monte Carlo return and value estimate) for improved training stability.
4.	Grid Search: • Tests combinations of learning rates, network sizes, update ratios, and random seeds to determine the best hyperparameters for each algorithm.

Project Structure
• gird_search.py: Contains grid search functions that tune hyperparameters for REINFORCE, AC, and A2C. CSV files and plots generated during grid search are saved in the “results” and “plots” directories, respectively.
• policy_networks.py: Implements the policy and value networks along with training functions for REINFORCE, Actor-Critic, and A2C. It also includes routines for plotting learning curves and saving output data.
• requirements.txt: Lists all required Python packages.
• results/: Directory where CSV results (rewards, steps, etc.) are automatically saved.
• plots/: Directory for generated graphs and learning curves.
Setup Instructions
1.	Install Python 3.7 or above.
2.	(Optional but recommended) Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate
3.	Navigate to the project directory.
4.	Install the required packages by running:
pip install -r requirements.txt

Running Experiments
The code is designed to execute one experiment at a time. To run an experiment, open the corresponding Python file and uncomment the relevant function call. For example, to run the grid search for all three algorithms, in gird_search.py you can uncomment:
if name == "main":
overall_start_time = time.time()
# Uncomment the experiment you want to run:
python gird_search.py  
# run_grid_search_a2c()         # A2C grid search  
# run_grid_search_reinforce()   # REINFORCE grid search  
# run_grid_search_actor_critic()# Actor-Critic grid search  
Similarly, to run the individual algorithms with default settings, in policy_networks.py uncomment the corresponding function call, then run:
python policy_networks.py
Output
After running an experiment, the following outputs will be automatically saved:
• Learning Curves: Smoothed plots of rewards over time in the “plots” directory.
• Final Performance Charts: Bar charts comparing the final rewards of different configurations/hyperparameters.
• CSV Files: Detailed logs of episode rewards and steps are saved in the “results” directory.
• Console Logs: Training progress and best hyperparameter combinations will be displayed during execution.


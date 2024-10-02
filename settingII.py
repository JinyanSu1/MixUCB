import numpy as np
from utils.plot import plot_average_rewards1, plot_cumulative_rewards1
from utils.run_simulation import run_simulation_mixucbII
from utils.linucb import initialize_ucb_algorithms
from utils.get_data import ContextGenerator
import matplotlib.pyplot as plt
import os
import argparse
import sys
np.random.seed(42)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run MixUCB simulation with different beta values.')
    parser.add_argument('--beta', type=float, default=2.0, help='Beta parameter for the logistic regression oracle')
    parser.add_argument('--temperature', type=float, default=0.1, help='expert action temperaure, lower temperature (close to 0) means he expert action is less noisy')
    parser.add_argument('--T', type=int, default=100, help='Number of rounds to run the simulation')
    parser.add_argument('--lambda_', type=float, default=0.001, help='regularization parameter for logistic regression')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate for the online lgistic regression oracle')
    parser.add_argument('--num_reps', type=int, default=1, help='Number of repetitions to run the simulation')
    parser.add_argument('--alpha', type=float, default=0.5, help='Alpha parameter for the LinUCB oracle')
    parser.add_argument('--delta', type=float, default=0.5, help='threshold for query')
    parser.add_argument('--reveal_reward', type=bool, default=True, help='set reveal_reward=True for setting II, and set it as False for setting I ')
    parser.add_argument('--n_actions', type=int, default=5, help='number of actions')
    # Parse command-line arguments
    args = parser.parse_args()

    # Use beta value from arguments
    beta = args.beta
    temperature = args.temperature
    num_reps = args.num_reps
    n_features = 2
    n_actions = args.n_actions
    noise_std = 0
    alpha_values = [args.alpha]
    delta = args.delta
    lambda_ = args.lambda_
    # parameter of logistic regression
    learning_rate = args.learning_rate
    reveal_reward = args.reveal_reward
    T = args.T

    colors = {
        'linucb': 'tab:purple',
        'mixucb': 'tab:blue',
        'always_query': 'tab:red'
    }
    Figure_dir = 'Figures/'
    Results_dir = 'Results/'
    if not os.path.exists(Results_dir):
        os.makedirs(Results_dir)
    if not os.path.exists(Figure_dir):
        os.makedirs(Figure_dir)
    log_file_path = os.path.join(Results_dir, f'print_output_alpha_beta{beta}_tmp{temperature}_lr{learning_rate}_numreps{num_reps}_T{T}_actions{n_actions}_delta_{delta}_lambda{lambda_}.txt')
    log_file = open(log_file_path, 'w')
    sys.stdout = log_file
    true_weights = np.random.randn(n_actions, n_features)
    norm = np.linalg.norm(true_weights)
    if norm > 1:  # To avoid division by zero
        true_weights = true_weights / norm
    generator = ContextGenerator(true_weights=true_weights, noise_std=noise_std, temperature=temperature)

    all_rewards = []
    labels = []
    query_data = []
    cumulative_rewards_linucb = []
    cumulative_rewards_mixucb = []
    cumulative_rewards_always_query = []

    cumulative_rewards_linucb_std = []
    cumulative_rewards_mixucb_std = []
    cumulative_rewards_always_query_std = []
    thetas_data = []
    covs_data = []
    # Create a grid of subplots

        


    for row_idx, alpha in enumerate(alpha_values):
        cumulative_rewards_mixucb_ = []
        cumulative_rewards_linucb_ = []
        cumulative_rewards_always_query_ = []
        for rep_id in range(num_reps):
            _, linucb, always_query_ucb, online_lr_oracle, online_sq_oracle = initialize_ucb_algorithms(n_actions, n_features, alpha, lambda_, learning_rate=learning_rate, beta= beta)

            # Run simulation for the current parameter
            cumulative_reward_mixucb, cumulative_reward_linucb, cumulative_reward_always_query, q, total_num_queries = run_simulation_mixucbII(
                T, delta, generator,  online_lr_oracle, online_sq_oracle, linucb, always_query_ucb, reveal_reward
            )
            cumulative_rewards_mixucb_.append(cumulative_reward_mixucb)
            cumulative_rewards_linucb_.append(cumulative_reward_linucb)
            cumulative_rewards_always_query_.append(cumulative_reward_always_query)


        
        cumulative_rewards_mixucb.append(np.mean(cumulative_rewards_mixucb_, axis=0))
        cumulative_rewards_linucb.append(np.mean(cumulative_rewards_linucb_, axis=0))
        cumulative_rewards_always_query.append(np.mean(cumulative_rewards_always_query_, axis=0))
        
        cumulative_rewards_linucb_std.append(np.std(cumulative_rewards_linucb_, axis=0))
        cumulative_rewards_mixucb_std.append(np.std(cumulative_rewards_mixucb_, axis=0))
        cumulative_rewards_always_query_std.append(np.std(cumulative_rewards_always_query_, axis=0))
            
        


    cumulative_rewards = {
        'LinUCB': cumulative_rewards_linucb,
        'MixUCB': cumulative_rewards_mixucb,
        'Always Query': cumulative_rewards_always_query
    }
    cumulative_rewards_std = {
        'LinUCB': cumulative_rewards_linucb_std,
        'MixUCB': cumulative_rewards_mixucb_std,
        'Always Query': cumulative_rewards_always_query_std
    }
    print('LinUCB', cumulative_rewards['LinUCB'])
    print('MixUCB', cumulative_rewards['MixUCB'])
    print('Always Query', cumulative_rewards['Always Query'])

    
    sys.stdout = sys.__stdout__
    log_file.close()
    # Create a 1 by m grid of subplots for average rewards
    fig, axs = plt.subplots(1, len(alpha_values), figsize=(5, 5))

    # Plot average rewards
    plot_average_rewards1(axs, cumulative_rewards, alpha_values, cumulative_rewards_std)
    plt.tight_layout()
    fig.savefig(os.path.join(Figure_dir, f'IIaverage_rewards_alpha_beta{beta}_tmp{temperature}_lr{learning_rate}_numreps{num_reps}_T{T}_actions{n_actions}_delta_{delta}test.jpg'), format='jpg', dpi=300, bbox_inches='tight')
    plt.show()

    fig, axs = plt.subplots(1, len(alpha_values), figsize=(5, 5))
    plot_cumulative_rewards1(axs, cumulative_rewards, alpha_values, cumulative_rewards_std)
    plt.tight_layout()
    fig.savefig(os.path.join(Figure_dir, f'IIcumulative_rewards_alpha_beta{beta}_tmp{temperature}_lr{learning_rate}_numreps{num_reps}_T{T}_actions{n_actions}_delta_{delta}test.jpg'), format='jpg', dpi=300, bbox_inches='tight')
    plt.show()
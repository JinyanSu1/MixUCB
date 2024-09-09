import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv
def run_simulation(T, delta, generator, mixucb, linucb, always_query_ucb, plot_rounds, action_plot):
    """Run the simulation and collect rewards and theta values."""
    cumulative_reward_mixucb = []
    cumulative_reward_linucb = []
    cumulative_reward_always_query = []
    reward_mixucb = 0
    reward_linucb = 0
    reward_always_query = 0
    q = np.zeros(T)
    total_num_queries = 0

    # Storage for plotting
    theta_data = {round_num: {'linucb': [], 'mixucb': [], 'always_query_ucb': []} for round_num in plot_rounds}
    cov_data = {round_num: {'linucb': [], 'mixucb': [], 'always_query_ucb': []} for round_num in plot_rounds}

    for i in range(T):
        context, true_rewards = generator.generate_context_and_rewards()
        mixucb_ucb, mixucb_lcb = mixucb.get_ucb_lcb(context)

        width = mixucb_ucb - mixucb_lcb
        action_hat = np.argmax(mixucb_ucb)
        width_Ahat = width[action_hat]

        if width_Ahat > delta:
            total_num_queries += 1
            expert_action = np.argmax(true_rewards)
            reward = true_rewards[expert_action]
            mixucb.update_all(context, true_rewards)
            q[i] = 1
        else:
            reward = true_rewards[action_hat]
            mixucb.update(action_hat, context, reward)
        reward_mixucb += reward
        cumulative_reward_mixucb.append(reward_mixucb)
        linucb_ucb, linucb_lcb = linucb.get_ucb_lcb(context)
        action_hat = np.argmax(linucb_ucb)
        reward = true_rewards[action_hat]
        linucb.update(action_hat, context, reward)
        reward_linucb += reward

        cumulative_reward_linucb.append(reward_linucb)
        expert_action = np.argmax(true_rewards)
        reward = true_rewards[expert_action]
        always_query_ucb.update_all(context, true_rewards)
        reward_always_query += reward

        cumulative_reward_always_query.append(reward_always_query)

        if i + 1 in plot_rounds:
            # Store theta and covariance data for plotting later
            theta_data[i + 1]['linucb'] = linucb.get_theta()[action_plot]
            theta_data[i + 1]['mixucb'] = mixucb.get_theta()[action_plot]
            theta_data[i + 1]['always_query_ucb'] = always_query_ucb.get_theta()[action_plot]
            cov_data[i + 1]['linucb'] = inv(linucb.A[action_plot])
            cov_data[i + 1]['mixucb'] = inv(mixucb.A[action_plot])
            cov_data[i + 1]['always_query_ucb'] = inv(always_query_ucb.A[action_plot])

    return cumulative_reward_mixucb, cumulative_reward_linucb, cumulative_reward_always_query, q, total_num_queries, theta_data, cov_data

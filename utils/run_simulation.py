import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv
import cvxpy as cp
from tqdm import tqdm
import time

class CBOptimization():
    def __init__(self, n_actions, dim, beta_sq, beta_lr):
        self.theta = cp.Variable((n_actions, dim))
        self.context = cp.Parameter(dim)
        self.theta_sq = cp.Parameter((n_actions, dim))
        self.As = [cp.Parameter((dim, dim), PSD=True) for _ in range(n_actions)]
        self.objectives = [cp.Maximize(cp.matmul(self.theta[a], self.context)) for a in range(n_actions)]
        self.constraints = [cp.quad_form(self.theta[a] - self.theta_sq[a], self.As[a]) <= beta_sq for a in range(n_actions)]

        self.theta_lr = cp.Parameter((n_actions, dim))
        self.X_sum = cp.Parameter((dim, dim), PSD=True)
        sum_quad_form = cp.sum([cp.quad_form(self.theta[a] - self.theta_lr[a], self.X_sum) for a in range(n_actions)])
  
        self.constraints.append(sum_quad_form <= beta_lr)

    def solve(self, context, theta_sq, theta_lr, As, X_sum, action, ucb=True):
        self.theta_sq.value = theta_sq
        self.theta_lr.value = theta_lr
        for A, AA in zip(self.As,As):
            A.value = AA
        self.X_sum.value = X_sum
        self.context.value = context

        prob = cp.Problem(self.objectives[action], self.constraints)
        prob.solve(solver='MOSEK')
        return prob.value



def solve_convex_optimization_ucb(obj_a, x_t, online_lr_oracle, online_sq_oracle, n_actions):
    x_t = x_t.flatten()
    theta = cp.Variable((n_actions, len(x_t)))  # Theta variables for each action
    objective = cp.Maximize(cp.matmul(theta[obj_a], x_t))  # Maximize theta_a^T x_t for a single action
    constraints = []
    theta_sq = online_sq_oracle.get_theta()
    A = [online_sq_oracle.A[a] for a in range(n_actions)]
    theta_lr, X_sum = online_lr_oracle.get_optimization_parameters()
    beta_sq = online_sq_oracle.alpha
    beta_lr = online_lr_oracle.beta
    # Add LinUCB constraint for each theta_a
    for a in range(n_actions):
        constraints.append(cp.quad_form(theta[a] - theta_sq[a], A[a]) <= beta_sq)

    # Add the logistic regression constraint
    sum_quad_form = cp.sum([cp.quad_form(theta[a] - theta_lr[a], X_sum) for a in range(n_actions)])
  
    constraints.append(sum_quad_form <= beta_lr)

    # Solve the optimization problem
    prob = cp.Problem(objective, constraints)
    prob.solve(solver='MOSEK')
    return prob.value
def solve_convex_optimization_lcb(obj_a, x_t, online_lr_oracle, online_sq_oracle, n_actions):
    x_t = x_t.flatten()
    theta = cp.Variable((n_actions, len(x_t)))  # Theta variables for each action
    objective = cp.Minimize(cp.matmul(theta[obj_a], x_t))  
    constraints = []
    theta_sq = online_sq_oracle.get_theta()
    A = [online_sq_oracle.A[a] for a in range(n_actions)]
    theta_lr, X_sum = online_lr_oracle.get_optimization_parameters()
    beta_sq = online_sq_oracle.alpha
    beta_lr = online_lr_oracle.beta
    # Add LinUCB constraint for each theta_a
    for a in range(n_actions):
        constraints.append(cp.quad_form(theta[a] - theta_sq[a], A[a]) <= beta_sq)

    # Add the logistic regression constraint
    sum_quad_form = cp.sum([cp.quad_form(theta[a] - theta_lr[a], X_sum) for a in range(n_actions)])
    constraints.append(sum_quad_form <= beta_lr)

    # Solve the optimization problem
    prob = cp.Problem(objective, constraints)
    prob.solve(solver='MOSEK')
    return prob.value
def run_simulation_mixucbIII(T, delta, generator, mixucb, linucb, always_query_ucb, plot_rounds, action_plot):
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
        context, true_rewards, _ = generator.generate_context_rewards_and_expert_action()
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
def run_simulation_mixucbII(T, delta, generator, online_lr_oracle, online_sq_oracle, linucb, always_query_ucb, reveal_reward= True):
    """Run the simulation and collect rewards and theta values."""
    cumulative_reward_mixucb = []
    cumulative_reward_linucb = []
    cumulative_reward_always_query = []
    reward_mixucb = 0
    reward_linucb = 0
    reward_always_query = 0
    q = np.zeros(T)
    total_num_queries = 0

    opt_prob = CBOptimization(generator.n_actions, generator.n_features, online_sq_oracle.alpha, online_lr_oracle.beta)

    
    for i in tqdm(range(T)):
        context, true_rewards, expert_action = generator.generate_context_rewards_and_expert_action()
        n_actions = len(true_rewards)
        actions_ucb = np.zeros(n_actions)
        currt = time.time()
        for obj_a in range(n_actions):
            # actions_ucb[obj_a] = solve_convex_optimization_ucb(obj_a, context, online_lr_oracle, online_sq_oracle, n_actions)
            theta_sq = online_sq_oracle.get_theta()
            As = [online_sq_oracle.A[a] for a in range(n_actions)]
            theta_lr, X_sum = online_lr_oracle.get_optimization_parameters()
            opt_prob.solve(context.flatten(), np.array(theta_sq), theta_lr, As, X_sum, obj_a, ucb=True)
            print(time.time()-currt)
            currt = time.time()
        action_hat = np.argmax(actions_ucb)
        currt=time.time()
        action_hat_lcb = solve_convex_optimization_lcb(action_hat, context, online_lr_oracle, online_sq_oracle, n_actions)
        print(time.time() - currt)
        width_Ahat = actions_ucb[action_hat] - action_hat_lcb


        if width_Ahat > delta:
            total_num_queries += 1
            
            online_lr_oracle.update(context, expert_action)
            q[i] = 1
  
            reward = true_rewards[expert_action]
            if reveal_reward:
                online_sq_oracle.update(expert_action, context, reward)
        else:
            reward = true_rewards[action_hat]
            online_sq_oracle.update(action_hat, context, reward)
        
        reward_mixucb += reward
        cumulative_reward_mixucb.append(reward_mixucb)
        linucb_ucb, linucb_lcb = linucb.get_ucb_lcb(context)
        action_hat = np.argmax(linucb_ucb)
        reward = true_rewards[action_hat]
        linucb.update(action_hat, context, reward)
        reward_linucb += reward

        cumulative_reward_linucb.append(reward_linucb)
        # expert_action = np.argmax(true_rewards)
        reward = true_rewards[expert_action]
        always_query_ucb.update_all(context, true_rewards)
        reward_always_query += reward

        cumulative_reward_always_query.append(reward_always_query)


    return cumulative_reward_mixucb, cumulative_reward_linucb, cumulative_reward_always_query, q, total_num_queries

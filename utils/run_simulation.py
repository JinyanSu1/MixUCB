import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv, sqrtm
import cvxpy as cp
from tqdm import tqdm
import torch
import sys
import time
import multiprocess as mp
import warnings

warnings.filterwarnings(action='ignore', message="You are solving a parameterized problem that is not DPP")

class CBOptimization():
    def __init__(self, n_actions, dim, beta_sq, beta_lr):
        self.n_actions = n_actions
        self.theta = cp.Variable((n_actions, dim))
        self.context = cp.Parameter(dim)
        self.theta_sq = cp.Parameter((n_actions, dim))
        self.As = [cp.Parameter((dim, dim), PSD=True) for _ in range(n_actions)]
        self.objectives_max = [cp.Maximize(cp.matmul(self.theta[a], self.context)) for a in range(n_actions)]
        self.objectives_min = [cp.Minimize(cp.matmul(self.theta[a], self.context)) for a in range(n_actions)]

        # LinUCB constraints
        self.constraints = [cp.quad_form(self.theta[a] - self.theta_sq[a], self.As[a]) <= beta_sq for a in range(n_actions)]

        # Logistic regression constraints
        self.theta_lr = cp.Parameter((n_actions, dim))
        self.X_sum = cp.Parameter((dim, dim), PSD=True)
        sum_quad_form = cp.sum([cp.quad_form(self.theta[a] - self.theta_lr[a], self.X_sum) for a in range(n_actions)])
        self.constraints.append(sum_quad_form <= beta_lr)
        self.constraints.append(cp.norm(self.theta, 'fro') <= 1)
        self.constraints.extend([cp.norm(self.theta[a], 'fro') <= 1 for a in range(n_actions)])
        self.problems_max = [cp.Problem(objmax, self.constraints) for objmax in self.objectives_max]
        self.problems_min = [cp.Problem(objmin, self.constraints) for objmin in self.objectives_min]
 
    def solve(self, context, theta_sq, theta_lr, As, X_sum, action, ucb=True):
        self.theta_sq.value = theta_sq
        self.theta_lr.value = theta_lr
        for A, AA in zip(self.As,As):
            A.value = AA
        self.X_sum.value = X_sum
        self.context.value = context

        if ucb:
            prob = self.problems_max[action]
        else:
            prob = self.problems_min[action]

        # TODO: add error checking
        prob.solve(solver='MOSEK')
        return prob.value

    def solve_allactions(self, context, theta_sq, theta_lr, As, X_sum, ucb=True):
        def solve_a(a):
            return self.solve(context, theta_sq, theta_lr, As, X_sum, a, ucb=ucb)

        pool = mp.Pool(processes = 1) # mp.cpu_count()-1)
        return pool.map(solve_a, list(range(self.n_actions)))



class CBOptimizationDPP():
    # to ensure DPP, replace quad_form with the following:
    # (theta - thetah)^T A (theta - thetah) = |A_sqrt theta|^2 - 2*theta^T (A @ thetah) + (thetah^T A thetah)
    def __init__(self, n_actions, dim, beta_sq, beta_lr):
        self.n_actions = n_actions
        self.theta = cp.Variable((n_actions, dim))
        self.context = cp.Parameter(dim)
        self.objectives_max = [cp.Maximize(cp.matmul(self.theta[a], self.context)) for a in range(n_actions)]
        self.objectives_min = [cp.Minimize(cp.matmul(self.theta[a], self.context)) for a in range(n_actions)]

        # LinUCB constraints
        self.As_sqrt = [cp.Parameter((dim, dim)) for _ in range(n_actions)]
        self.Astheta_sq = [cp.Parameter(dim) for _ in range(n_actions)]
        self.quadAstheta_sq = [cp.Parameter() for _ in range(n_actions)]
        self.constraints = [cp.sum_squares(self.As_sqrt[a] @ self.theta[a]) - 2*cp.matmul(self.theta[a], self.Astheta_sq[a]) \
                            + self.quadAstheta_sq[a]  <= beta_sq for a in range(n_actions)]

        # Logistic regression constraints
        self.X_sum_sqrt = cp.Parameter((dim, dim))
        self.X_sum_theta_lr = [cp.Parameter(dim) for _ in range(n_actions)]
        self.quadX_sum_theta_lr = [cp.Parameter() for _ in range(n_actions)]
        sum_quad_form = cp.sum([cp.sum_squares(self.X_sum_sqrt @ self.theta[a]) - 2*cp.matmul(self.theta[a], self.X_sum_theta_lr[a]) \
                                + self.quadX_sum_theta_lr[a] for a in range(n_actions)])
        self.constraints.append(sum_quad_form <= beta_lr)
        self.constraints.append(cp.norm(self.theta, 'fro') <= 1)
        self.constraints.extend([cp.norm(self.theta[a], 'fro') <= 1 for a in range(n_actions)])
        self.problems_max = [cp.Problem(objmax, self.constraints) for objmax in self.objectives_max]
        self.problems_min = [cp.Problem(objmin, self.constraints) for objmin in self.objectives_min]

        self.precompiled_max = [prob.get_problem_data(cp.MOSEK) for prob in self.problems_max]
        self.precompiled_min = [prob.get_problem_data(cp.MOSEK) for prob in self.problems_min]
        
 
    def solve(self, context, theta_sq, theta_lr, As, As_sqrt, X_sum, X_sum_sqrt, action, ucb=True):
        # TODO: make it so all optimizatino probelms are actually the same
        for A, AA in zip(self.As_sqrt,As_sqrt):
            A.value = AA
        for At, A, t in zip(self.Astheta_sq,As_sqrt,theta_sq):
            At.value = A @ t
        for tAt, At, t in zip(self.quadAstheta_sq,self.Astheta_sq,theta_sq):
            tAt.value = t @ At.value

        self.X_sum_sqrt.value = X_sum_sqrt
        for Xt, t in zip(self.X_sum_theta_lr, theta_lr):
            Xt.value = X_sum @ t
        for tXt, Xt, t in zip(self.quadX_sum_theta_lr, self.X_sum_theta_lr, theta_lr):
            tXt.value = t @ Xt.value

        self.context.value = context

       
        if ucb:
            prob = self.problems_max[action]
            # data, chain, inverse_data = self.precompiled_max[action]
        else:
            prob = self.problems_min[action]
            # data, chain, inverse_data = self.precompiled_min[action]

        # soln = chain.solve_via_data(prob, data)
        # unpacks the solution returned by SCS into `problem`
        # prob.unpack_results(soln, chain, inverse_data)
        
        # TODO: add error checking
        prob.solve(solver='MOSEK') #, verbose=(not ucb))
        return prob.value

    def solve_allactions(self, context, theta_sq, theta_lr, As, As_sqrt, X_sum, X_sum_sqrt, 
                         ucb=True, multithreading=True):
        def solve_a(a):
            return self.solve(context, theta_sq, theta_lr, As, As_sqrt, X_sum, X_sum_sqrt, a, ucb=ucb)

        if multithreading:
            pool = mp.Pool(processes = 1) # mp.cpu_count()-1)
            ret = pool.map(solve_a, list(range(self.n_actions)))
        else: 
            ret = []
            for a in range(self.n_actions):
                ret.append(solve_a(a))
        return ret 




def solve_convex_optimization_ucb(obj_a, x_t, online_lr_oracle, online_sq_oracle, n_actions):
    x_t = x_t.flatten()
    theta = cp.Variable((n_actions, len(x_t)))  # Theta variables for each action
    objective = cp.Maximize(cp.matmul(theta[obj_a], x_t))  # Maximize theta_a^T x_t for a single action
    constraints = []
    theta_sq = online_sq_oracle.get_theta()
    A = [online_sq_oracle.A[a] for a in range(n_actions)]
    theta_lr, X_sum = online_lr_oracle.get_optimization_parameters()
    beta_sq = online_sq_oracle.alpha**2
    beta_lr = online_lr_oracle.beta
    # Add LinUCB constraint for each theta_a
    for a in range(n_actions):
        constraints.append(cp.quad_form(theta[a] - theta_sq[a], A[a]) <= beta_sq)
        constraints.append(cp.norm(theta[a], 2) <= 1)

    # Add the logistic regression constraint
    sum_quad_form = cp.sum([cp.quad_form(theta[a] - theta_lr[a], X_sum) for a in range(n_actions)])
  
    constraints.append(sum_quad_form <= beta_lr)

    # Solve the optimization problem
    prob = cp.Problem(objective, constraints)
    prob.solve(solver='MOSEK')
    # print("Value of sum_quad_form:", sum_quad_form.value)
    sys.stdout.flush()
    return prob.value
def solve_convex_optimization_lcb(obj_a, x_t, online_lr_oracle, online_sq_oracle, n_actions):
    x_t = x_t.flatten()
    theta = cp.Variable((n_actions, len(x_t)))  # Theta variables for each action
    objective = cp.Minimize(cp.matmul(theta[obj_a], x_t))  
    constraints = []
    theta_sq = online_sq_oracle.get_theta()
    A = [online_sq_oracle.A[a] for a in range(n_actions)]
    theta_lr, X_sum = online_lr_oracle.get_optimization_parameters()
    beta_sq = online_sq_oracle.alpha**2
    beta_lr = online_lr_oracle.beta
    # Add LinUCB constraint for each theta_a
    for a in range(n_actions):
        constraints.append(cp.quad_form(theta[a] - theta_sq[a], A[a]) <= beta_sq)
        constraints.append(cp.norm(theta[a], 2) <= 1)

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

    # opt_prob = CBOptimization(generator.n_actions, generator.n_features, online_sq_oracle.alpha**2, online_lr_oracle.beta)
    #opt_probDPP = CBOptimizationDPP(generator.n_actions, generator.n_features, online_sq_oracle.alpha**2, online_lr_oracle.beta)
    
    for i in tqdm(range(T)):
        print('current round:', i)
        context, true_rewards, expert_action = generator.generate_context_rewards_and_expert_action()
        n_actions = len(true_rewards)
        actions_ucb = np.zeros(n_actions)
        for obj_a in range(n_actions):
            actions_ucb[obj_a] = solve_convex_optimization_ucb(obj_a, context, online_lr_oracle, online_sq_oracle, n_actions)
            
        action_hat = np.argmax(actions_ucb)
        action_hat_lcb = solve_convex_optimization_lcb(action_hat, context, online_lr_oracle, online_sq_oracle, n_actions)
        # print('actions_ucb:(Jinyan)', actions_ucb)
        # print('action_hat_lcb:(Jinyan)', action_hat_lcb)

        # As = [online_sq_oracle.A[a] for a in range(n_actions)]
        # theta_sq = online_sq_oracle.get_theta()
        # theta_lr, X_sum = online_lr_oracle.get_optimization_parameters()
        # currt = time.time()
        # As_sqrt = [sqrtm(A) for A in As]
        # X_sum_sqrt = sqrtm(X_sum)
        # print('sqrt', time.time()-currt)

        # # Not DPP
        # currt = time.time()
        # actions_ucb = opt_prob.solve_allactions(context.flatten(), np.array(theta_sq), theta_lr, As, X_sum)
        # print(time.time()-currt, (time.time()-currt)/n_actions)
        # action_hat = np.argmax(actions_ucb)
        
        # currt=time.time()
        # action_hat_lcb = opt_prob.solve(context.flatten(), np.array(theta_sq), theta_lr, As, X_sum, action_hat, ucb=False)
        # print(time.time() - currt)
        # width_Ahat = actions_ucb[action_hat] - action_hat_lcb
        # print('width_Ahat:', width_Ahat)

        # DPP
        # currt = time.time()
        # actions_ucb = opt_probDPP.solve_allactions(context.flatten(), np.array(theta_sq), theta_lr, 
        #                                            As, As_sqrt, X_sum, X_sum_sqrt, multithreading=False)
        # print('DPP', time.time()-currt, (time.time()-currt)/n_actions)
        # action_hat = np.argmax(actions_ucb)
        
        # currt=time.time()
        # action_hat_lcb = opt_probDPP.solve(context.flatten(), np.array(theta_sq), theta_lr, As, As_sqrt, X_sum, X_sum_sqrt, action_hat, ucb=False)
        # print(time.time() - currt)
        width_Ahat = actions_ucb[action_hat] - action_hat_lcb
        # print('actions_ucb:(Sarah)', actions_ucb)
        # print('action_hat_lcb:(Sarah)', action_hat_lcb)
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
        # print('reward_mixucb:', reward_mixucb, 'query:', q[i])
        
 

        
        
        
        
        
        
        
        
        linucb_ucb, linucb_lcb = linucb.get_ucb_lcb(context)
        action_hat = np.argmax(linucb_ucb)
        reward = true_rewards[action_hat]
        linucb.update(action_hat, context, reward)
        reward_linucb += reward

        cumulative_reward_linucb.append(reward_linucb)
        # print('reward_linucb:', reward_linucb)
        # expert_action = np.argmax(true_rewards)
        reward = true_rewards[expert_action]
        always_query_ucb.update_all(context, true_rewards)
        reward_always_query += reward

        cumulative_reward_always_query.append(reward_always_query)

        # print('reward_always_query:', reward_always_query)
    return cumulative_reward_mixucb, cumulative_reward_linucb, cumulative_reward_always_query, q, total_num_queries


def run_simulation_mixucbII_test_lr(T, delta, generator, online_lr_oracle, online_sq_oracle, linucb, always_query_ucb, reveal_reward= True):
    """Run the simulation and collect rewards and theta values."""

    lr_convergence_errors = []
    lr_convergence_errors_linucb = []
    true_parameters = generator.true_weights  # Assuming true weights are stored in the generator

    
    for i in tqdm(range(T)):
        context, true_rewards, expert_action = generator.generate_context_rewards_and_expert_action()
        online_lr_oracle.update(context, expert_action)

  



        # Calculate logistic regression parameter error (convergence test)
        learned_parameters = online_lr_oracle.get_model_params()
        error = np.linalg.norm(learned_parameters - true_parameters)
        lr_convergence_errors.append(error)
        

        
        
        # LinUCB action selection and updating
        linucb_ucb, linucb_lcb = linucb.get_ucb_lcb(context)
        action_hat = np.argmax(linucb_ucb)
        reward = true_rewards[action_hat]
        linucb.update(action_hat, context, reward)
        learned_parameters_linucb = linucb.get_theta()
        error_linucb = np.linalg.norm(learned_parameters_linucb - true_parameters)
        lr_convergence_errors_linucb.append(error_linucb)
    return lr_convergence_errors, lr_convergence_errors_linucb

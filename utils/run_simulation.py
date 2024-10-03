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

        self.constraints.extend([cp.norm(self.theta[a], 2) <= 1 for a in range(n_actions)])

        self.problems_max = [cp.Problem(objmax, self.constraints) for objmax in self.objectives_max]
        self.problems_min = [cp.Problem(objmin, self.constraints) for objmin in self.objectives_min]

        # this step is necessary for speedups!
        self.precompiled_max = [prob.get_problem_data(cp.MOSEK) for prob in self.problems_max]
        self.precompiled_min = [prob.get_problem_data(cp.MOSEK) for prob in self.problems_min]
        
 
    def solve(self, context, theta_sq, theta_lr, As, As_sqrt, X_sum, X_sum_sqrt, action, ucb=True):
        for A, AA in zip(self.As_sqrt,As_sqrt):
            A.value = AA
        for At, A, t in zip(self.Astheta_sq,As,theta_sq):
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
        else:
            prob = self.problems_min[action]

        # this is not necessary for speedup
        # data, chain, inverse_data = self.precompiled_min[action]
        # soln = chain.solve_via_data(prob, data)
        # prob.unpack_results(soln, chain, inverse_data)
        
        # TODO: add error checking/catching
        prob.solve(solver='MOSEK')
        print('prob.value:', prob.value)
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


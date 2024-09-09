import numpy as np
from scipy.linalg import inv
class LinUCB:
    def __init__(self, n_actions, n_features, alpha=1.0, lambda_=1.0):
        self.n_actions = n_actions
        self.n_features = n_features
        self.alpha = alpha
        self.lambda_ = lambda_
        self.A = [lambda_ * np.identity(n_features) for _ in range(n_actions)]
        self.b = [np.zeros(n_features) for _ in range(n_actions)]

    def update(self, action, context, reward):
        context = context.reshape(-1)
        self.A[action] += np.outer(context, context)
        self.b[action] += reward * context

    def update_all(self, context, rewards):
        context = context.reshape(-1)
        for a in range(self.n_actions):
            self.A[a] += np.outer(context, context)
            self.b[a] += rewards[a] * context

    def get_theta(self):
        return [inv(self.A[a]).dot(self.b[a]) for a in range(self.n_actions)]

    def get_ucb_lcb(self, context):
        context = context.reshape(-1)
        ucb = []
        lcb = []
        for a in range(self.n_actions):
            theta_a = inv(self.A[a]).dot(self.b[a])
            sigma_a = self.alpha * np.sqrt(context.dot(inv(self.A[a]).dot(context)))
            ucb.append(theta_a.dot(context) + sigma_a)
            lcb.append(theta_a.dot(context) - sigma_a)
        return np.array(ucb), np.array(lcb)

def initialize_ucb_algorithms(n_actions, n_features, alpha, lambda_):
    """Initialize UCB algorithms."""
    mixucb = LinUCB(n_actions, n_features, alpha, lambda_)
    linucb = LinUCB(n_actions, n_features, alpha, lambda_)
    always_query_ucb = LinUCB(n_actions, n_features, alpha, lambda_)
    return mixucb, linucb, always_query_ucb


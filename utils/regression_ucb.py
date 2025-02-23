import numpy as np
from scipy.linalg import inv
from sklearn.linear_model import SGDClassifier, LogisticRegression
from icecream import ic
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings

class MixUCB:
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
        # ic(self.b, reward, context)

    def update_all(self, context, rewards):
        context = context.reshape(-1)
        for a in range(self.n_actions):
            self.A[a] += np.outer(context, context)
            self.b[a] += rewards[a] * context

    def get_theta(self):
        # ic(self.A, self.b)
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


class CombinedLinearModel(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim, output_dim, lr=0.01, weight_decay=0.01, epochs=1000, tol=1e-4):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.weight_decay = weight_decay # l2 reg times 2
        self.epochs = epochs
        self.model = nn.Linear(input_dim, output_dim, bias=False)
        self.criterion_log = nn.CrossEntropyLoss()
        self.criterion_sq = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.coef_ = np.zeros([output_dim,input_dim])
        self.tolerance = tol
    
    def fit(self, X_log, y_log, X_sq=None, y_sq=None, verbose=False):
        if len(X_log)>0 and len(y_log)>0:
            X_log_tensor = torch.tensor(X_log, dtype=torch.float32)
            y_log_tensor = torch.tensor(y_log, dtype=torch.long)

        if X_sq is not None and len(X_sq)>0:
            X_sq_tensor = torch.tensor(X_sq, dtype=torch.float32)
            non_nan_mask = ~np.isnan(y_sq)
            y_sq_tensor = torch.tensor(y_sq[non_nan_mask], dtype=torch.float32)

        previous_loss = float('inf')
        
        for epoch in range(self.epochs):
            loss_log = 0
            if len(X_log)>0 and len(y_log)>0:
                outputs_log = self.model(X_log_tensor)
                loss_log = self.criterion_log(outputs_log, y_log_tensor)

            loss_sq = 0
            if X_sq is not None and len(X_sq)>0:
                outputs_sq = self.model(X_sq_tensor)
                loss_sq = self.criterion_sq(outputs_sq[non_nan_mask], y_sq_tensor)

            loss = loss_log + loss_sq
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Check for convergence
            if abs(previous_loss - loss.item()) < self.tolerance:
                if verbose:
                    print(f'Convergence reached at epoch {epoch+1}')
                break
            
            previous_loss = loss.item()

            if (epoch + 1) % 10 == 0 and verbose:
                print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}')
        if epoch == self.epochs:
            warnings.warn("Convergence criteria were not met.", UserWarning)
        self.coef_ = self.model.weight.detach().numpy()
    
    def predict(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        outputs = self.model(X_tensor)
        _, predicted = torch.max(outputs, 1)
        return predicted.numpy()
    
    def predict_proba(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        outputs = self.model(X_tensor)
        return torch.softmax(outputs, dim=1).detach().numpy()

class OnlineLogisticRegressionOracle:
    def __init__(self, n_features, n_actions, learning_rate, reg_coeff, rad_log, rad_sq=None, max_epochs=1000, tol=1e-4):
        self.model = CombinedLinearModel(n_features, n_actions, lr=learning_rate, weight_decay=reg_coeff/2, epochs=max_epochs, tol=tol)
        self.n_actions = n_actions
        self.n_features = n_features
        self.lambda_ = reg_coeff
        self.X_sum = self.lambda_/2 * np.eye(n_features)  # Accumulated X^T X
        self.A_sum = [self.lambda_/2 * np.eye(n_features) for _ in range(n_actions)]
        self.beta_log = rad_log
        self.beta_sq = rad_sq if rad_sq is not None else rad_log
        self.beta = self.beta_log # todo revmove
        self.Xs = []
        self.ys = []
        self.rs = []
        self.ind_log = []
        self.ind_sq = []

    def update(self, context, action=None, reward=None, rewards=None):
        self.Xs.append(context.ravel())
        self.ys.append(action)
        self.ind_log.append((action is not None) and reward is None and rewards is None)
        if reward is not None:
            assert rewards is None and action is not None
            rewards = np.nan * np.ones(self.n_actions)
            rewards[action] = reward
        self.ind_sq.append((rewards is not None))
        if rewards is None: rewards = np.nan * np.ones(self.n_actions)
        self.rs.append(rewards)

        self.model.fit(np.array(self.Xs)[self.ind_log], np.array(self.ys)[self.ind_log], np.array(self.Xs)[self.ind_sq], np.array(self.rs)[self.ind_sq])
        
        xxT = np.outer(context.ravel(), context.ravel())
        if action is not None:
            self.X_sum += xxT
        if rewards is not None:
            for a, r in enumerate(rewards):
                if not np.isnan(r):
                    self.A_sum[a] += xxT

    def get_model_params(self):
        return self.model.coef_

    def predict(self, X):
        # picks action
        return self.model.predict(X)

    def get_optimization_parameters(self):
        theta = self.get_model_params()  
        return theta, self.X_sum, self.A_sum

    def get_ucb_lcb(self, context):
        context = context.reshape(-1)
        ucb = []
        lcb = []
        theta, X_sum, A_sum = self.get_optimization_parameters()
        # TODO also implement separate CI
        combined_cov = [X_sum / self.beta_log**2 + A / self.beta_sq**2 for A in A_sum]
        for a in range(self.n_actions):
            sigma = np.sqrt(context.dot(inv(combined_cov[a]).dot(context)))
            ucb.append(theta[a].dot(context) + sigma)
            lcb.append(theta[a].dot(context) - sigma)
        return np.array(ucb), np.array(lcb)

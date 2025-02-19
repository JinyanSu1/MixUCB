import numpy as np
from scipy.linalg import inv
from sklearn.linear_model import SGDClassifier, LogisticRegression
from icecream import ic
import argparse

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

def initialize_ucb_algorithms(n_actions, n_features, alpha, lambda_, learning_rate = 1.0, beta= 1.0):
    """Initialize UCB algorithms."""
    mixucb = LinUCB(n_actions, n_features, alpha, lambda_)
    linucb = LinUCB(n_actions, n_features, alpha, lambda_)
    always_query_ucb = LinUCB(n_actions, n_features, alpha, lambda_)
    online_lr_oracle = OnlineLogisticRegressionOracle(n_features, n_actions, learning_rate, lambda_, beta)
    online_sq_oracle= LinUCB(n_actions, n_features, alpha, lambda_)
    return mixucb, linucb, always_query_ucb, online_lr_oracle, online_sq_oracle


import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin


class TorchLogReg(nn.Module): 
    def __init__(self, input_size, num_classes): 
        super(LogisticRegression, self).__init__() 
        self.linear = nn.Linear(input_size, num_classes, bias=False) 
        self.criterion = nn.CrossEntropyLoss() 
  
    def forward(self, x): 
        out = self.linear(x) 
        out = nn.functional.softmax(out, dim=1) 
        return out 

class MaxEntLinearModel(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim, output_dim, lr=0.01, weight_decay=0.01, epochs=100, tol=1e-4):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.weight_decay = weight_decay # l2 reg
        self.epochs = epochs
        self.model = nn.Linear(input_dim, output_dim)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.coef_ = np.zeros([output_dim,input_dim])
        self.tolerance = tol
    
    def fit(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        
        previous_loss = float('inf')
        
        for epoch in range(self.epochs):
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Check for convergence
            # if abs(previous_loss - loss.item()) < self.tolerance:
            #     print(f'Convergence reached at epoch {epoch+1}')
            #     break
            
            previous_loss = loss.item()

            # if (epoch + 1) % 10 == 0:
            #     print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}')
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
    def __init__(self, n_features, n_actions, learning_rate=0.1, lambda_=1.0, beta=1.0,
                 mode="sgd"):
        self.mode = mode
        if self.mode == "sgd":
            self.model = SGDClassifier(loss='log_loss', learning_rate='constant', eta0=learning_rate, alpha=lambda_, fit_intercept=False)  # Multi-class logistic regression
        elif self.mode == "logreg":
            self.model = LogisticRegression(C=1/lambda_, fit_intercept=False)  # Multi-class logistic regression multi_class='multinomial', 
        elif self.mode == "torch":
            # self.model = LogisticRegression(input_size=n_features, num_classes=n_actions) 
            # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
            self.model = MaxEntLinearModel(n_features, n_actions, lr=learning_rate, weight_decay=lambda_, epochs=100, tol=1e-4)
        else:
            raise Exception("invalid mode for logistic regression oracle")
        self.n_actions = n_actions
        self.n_features = n_features
        self.X_sum = np.zeros((n_features, n_features))  # Accumulated X^T X
        self.lambda_ = lambda_
        self.beta = beta
        self.Xs = []
        self.ys = []
        # # Initialize the model with some dummy data to set the number of classes (multi-class)
        dummy_X = np.zeros((n_actions, n_features))
        dummy_y = np.array([i for i in range(n_actions)])  # Dummy class label
        if self.mode == "sgd":
            self.model.partial_fit(dummy_X, dummy_y, classes=np.arange(n_actions))  # Initialize multi-class model
        elif self.mode == "logreg":
            self.model.fit(dummy_X, dummy_y)  # Initialize multi-class model

    def update(self, x_t, action):
        x_t_flat = x_t.ravel()
        self.Xs.append(x_t_flat)
        self.ys.append(action)
        # Update the logistic regression model with the new data point
        # if self.mode == "torch":
        #     # Define training parameters 
        #     num_iters = 1000
              
        #     # Train the model 
        #     for i in range(num_iters): 
        #         # Forward pass 
        #         outputs = self.model(self.Xs) 
        #         loss = self.model.criterion(outputs, self.ys) 
          
        #         # Backward and optimize 
        #         self.optimizer.zero_grad() 
        #         loss.backward() 
        #         self.optimizer.step() 
        if self.mode == "torch":
            self.model.fit(self.Xs, self.ys)
        elif len(np.unique(self.ys)) == self.n_actions:
            self.model.fit(self.Xs, self.ys)
        elif self.mode == "sgd":
            self.model.partial_fit(self.Xs, self.ys)
        self.X_sum += np.outer(x_t_flat, x_t_flat)  # Update X^T X sum for logistic regression constraint

    def get_model_params(self):
        # Return the parameter vector for each class (action)
        # if self.mode == "torch":
        #     return linear_layer.weight.data
        if self.n_actions == 2 and (self.mode in ['sgd','logreg']):
            return np.vstack([self.model.coef_, -1*self.model.coef_])
        return self.model.coef_

    def predict(self, X):
        # X is num samples by num features
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
        
    def get_optimization_parameters(self):
        """
        Returns the parameters required for the convex optimization:
        - theta_lr: Logistic regression parameters for each action.
        - X_sum: the accumulated X^T X matrix (with regularization).
        """
        theta_lr = self.get_model_params()  # Get the logistic regression model's coefficients (theta)
        #  X_sum, with regularization

        X_sum = self.X_sum + np.eye(self.n_features) * self.lambda_
        return theta_lr, X_sum

    def get_ucb_lcb(self, context):
        # this is valid when only considering logistic regression data
        context = context.reshape(-1)
        ucb = []
        lcb = []
        theta, X_sum = self.get_optimization_parameters()
        sigma = self.beta * np.sqrt(context.dot(inv(X_sum).dot(context)))
        for a in range(self.n_actions):
            ucb.append(theta[a].dot(context) + sigma)
            lcb.append(theta[a].dot(context) - sigma)
        return np.array(ucb), np.array(lcb)

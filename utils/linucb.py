import numpy as np
from scipy.linalg import inv
from sklearn.linear_model import SGDClassifier

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

def initialize_ucb_algorithms(n_actions, n_features, alpha, lambda_, learning_rate = 1.0, beta= 1.0):
    """Initialize UCB algorithms."""
    mixucb = LinUCB(n_actions, n_features, alpha, lambda_)
    linucb = LinUCB(n_actions, n_features, alpha, lambda_)
    always_query_ucb = LinUCB(n_actions, n_features, alpha, lambda_)
    online_lr_oracle = OnlineLogisticRegressionOracle(n_features, n_actions, learning_rate, lambda_, beta)
    online_sq_oracle= LinUCB(n_actions, n_features, alpha, lambda_)
    return mixucb, linucb, always_query_ucb, online_lr_oracle, online_sq_oracle

class OnlineLogisticRegressionOracle:
    def __init__(self, n_features, n_actions, learning_rate=0.1, lambda_=1.0, beta=1.0):
        self.model = SGDClassifier(loss='log_loss', learning_rate='constant', eta0=learning_rate, alpha=lambda_)  # Multi-class logistic regression
        self.n_actions = n_actions
        self.n_features = n_features
        self.X_sum = np.zeros((n_features, n_features))  # Accumulated X^T X
        self.lambda_ = lambda_
        self.beta = beta
        # Initialize the model with some dummy data to set the number of classes (multi-class)
        dummy_X = np.zeros((1, n_features))
        dummy_y = np.array([0])  # Dummy class label
        self.model.partial_fit(dummy_X, dummy_y, classes=np.arange(n_actions))  # Initialize multi-class model

    def update(self, x_t, action):
        x_t_flat = x_t.ravel()
        # Update the logistic regression model with the new data point
        self.model.partial_fit([x_t_flat], [action])
        self.X_sum += np.outer(x_t_flat, x_t_flat)  # Update X^T X sum for logistic regression constraint

    def get_model_params(self):
        # Return the parameter vector for each class (action)
        return self.model.coef_

    def predict(self, x_t):
        x_t_flat = x_t.ravel()
        # Return the predicted probabilities for each class
        return self.model.predict_proba([x_t_flat])
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


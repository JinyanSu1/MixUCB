
import numpy as np
## synthetic data
class ContextGenerator:
    def __init__(self, true_weights, use_softmax=True, noise_std=0.0):
        self.use_softmax = use_softmax
        self.noise_std = noise_std
        self.true_weights = true_weights
        self.n_actions, self.n_features = true_weights.shape

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, keepdims=True))  # subtract max for numerical stability
        return e_x / e_x.sum(keepdims=True)
    
    def generate_context(self):
        context = np.random.randn(1, self.n_features)
        return context
    
    def get_reward(self, context):
        logits = np.dot(context, self.true_weights.T)
        if self.use_softmax:
            rewards = self.softmax(logits.reshape(1, -1)).flatten()
        else:
            rewards = logits.flatten() + self.noise_std * np.random.randn(self.n_actions)
        return rewards

    def generate_context_and_rewards(self):
        context = self.generate_context()
        rewards = self.get_reward(context)
        return context, rewards
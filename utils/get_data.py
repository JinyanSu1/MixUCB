
import numpy as np
## synthetic data
class ContextGenerator:
    def __init__(self, true_weights, noise_std=0.0):
        
        self.noise_std = noise_std
        self.true_weights = true_weights
        self.n_actions, self.n_features = true_weights.shape
    def generate_context_and_rewards(self):
        """
        returns a single context and true rewards corresponding to each action.

        context, rewards, noiseless_rewards
        """
        # sample random context in unit circle.
        context = np.random.randn(1, self.n_features)
        norm = np.linalg.norm(context)
        if norm > 1:  
            context = context / norm
        noiseless_rewards = np.dot(context, self.true_weights.T)
        logits = noiseless_rewards + self.noise_std * np.random.randn(self.n_actions)
        noiseless_rewards = noiseless_rewards.flatten()
        rewards = logits.flatten()
        return context, rewards, noiseless_rewards

        
    




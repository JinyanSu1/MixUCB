
import numpy as np
## synthetic data
class ContextGenerator:
    def __init__(self, true_weights, noise_std=0.0, temperature = 1.0):
        
        self.noise_std = noise_std
        self.true_weights = true_weights
        self.n_actions, self.n_features = true_weights.shape
        self.temperature = temperature

    def softmax(self, x, temp):
        
        x_scaled = x / temp
        e_x = np.exp(x_scaled - np.max(x_scaled, keepdims=True))  # subtract max for numerical stability
        return e_x / e_x.sum(keepdims=True)
    
    def generate_context(self):
        context = np.random.randn(1, self.n_features)
        return context
    
    def get_reward(self, context):
        logits = np.dot(context, self.true_weights.T) + self.noise_std * np.random.randn(self.n_actions)
        rewards = self.softmax(logits.flatten() , temp=1)
        return rewards

    def generate_context_and_rewards(self):
        context = self.generate_context()
        rewards = self.get_reward(context)
        return context, rewards
    def generate_context_rewards_and_expert_action(self):
        context, rewards = self.generate_context_and_rewards()
        action_probs = self.softmax(rewards, temp=self.temperature)
        noisy_action = np.random.choice(self.n_actions, p=action_probs)
        return context, rewards, noisy_action
        
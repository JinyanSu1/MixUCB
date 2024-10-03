import numpy as np
import pickle
from utils.get_data import ContextGenerator
import argparse

import numpy as np
import pickle
from utils.get_data import ContextGenerator
import argparse

def generate_data(T, n_actions, n_features, noise_std, seed):
    # Set random seed for reproducibility
    np.random.seed(seed)

    # Generate true weights (theta) for actions
    true_weights = np.random.randn(n_actions, n_features)
    norm = np.linalg.norm(true_weights)
    if norm > 1:
        true_weights = true_weights / norm

    # Initialize context generator
    generator = ContextGenerator(true_weights=true_weights, noise_std=noise_std)

    # Store data for each round
    data = {
        "true_theta": true_weights,
        "rounds": []
    }

    # Generate data for T rounds
    for t in range(T):
        context, true_rewards = generator.generate_context_and_rewards()
        
        # Store context, true_rewards, and expert_action for each round
        data["rounds"].append({
            "context": context,
            "true_rewards": true_rewards,
        })

    return data

if __name__ == "__main__":
    # Argument parser for setting T, n_actions, n_features, and other parameters
    parser = argparse.ArgumentParser(description='Generate Data for T rounds and store in a pickle file')
    parser.add_argument('--T', type=int, default=1000, help='Number of rounds to generate')
    parser.add_argument('--n_actions', type=int, default=30, help='Number of actions')
    parser.add_argument('--n_features', type=int, default=2, help='Number of features for each context')
    parser.add_argument('--noise_std', type=float, default=0, help='Noise standard deviation for reward generation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--output_file', type=str, default='simulation_data.pkl', help='Output pickle file to store the data')
    
    args = parser.parse_args()

    # Generate the data
    data = generate_data(T=args.T, n_actions=args.n_actions, n_features=args.n_features, noise_std=args.noise_std, seed=args.seed)

    # Save data to a pickle file
    with open(args.output_file, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Data for {args.T} rounds generated and saved to {args.output_file} with seed {args.seed}")

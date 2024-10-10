## 10/9/24
## Toy example: 2d context, 2 arms.
## For sanity-checking linucb.
from utils.get_data import ContextGenerator
import argparse
import pickle as pkl
import numpy as np

# Argument parser for setting T, n_actions, n_features, and other parameters
parser = argparse.ArgumentParser(description='Generate Data for T rounds and store in a pickle file')
parser.add_argument('--T', type=int, default=1000, help='Number of rounds to generate')
# parser.add_argument('--n_actions', type=int, default=4, help='Number of actions')
# parser.add_argument('--n_features', type=int, default=4, help='Number of features for each context')
parser.add_argument('--noise_std', type=float, default=0, help='Noise standard deviation for reward generation')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--output_file', type=str, default='simulation_data_toy20241009.pkl', help='Output pickle file to store the data')

def generate_data(T, n_actions, n_features, noise_std, seed):
    # Set random seed for reproducibility
    # np.random.seed(seed)

    # Generate true weights (theta) for actions
    # 2 classes behaves weirdly with OnlineLogisticRegressionOracle, so switching to 3.
    # true_weights = np.array([[1/np.sqrt(2),1/np.sqrt(2)],[-1/np.sqrt(2),-1/np.sqrt(2)]])
    # true_weights = np.array([[1/np.sqrt(2),1/np.sqrt(2)],[-1/np.sqrt(2),-1/np.sqrt(2)]])
    true_weights = np.array([[np.cos(0),np.sin(0)],[np.cos(2/3*np.pi),np.sin(2/3*np.pi)],[np.cos(4/3*np.pi),np.sin(4/3*np.pi)]])

    # Initialize context generator
    generator = ContextGenerator(true_weights=true_weights, noise_std=noise_std)

    # Store data for each round
    data = {
        "true_theta": true_weights,
        "rounds": []
    }

    # Generate data for T rounds
    for t in range(T):
        context, noisy_rewards, noiseless_rewards = generator.generate_context_and_rewards()
        
        # Store context, true_rewards, and expert_action for each round
        data["rounds"].append({
            "context": context,
            "true_rewards": noisy_rewards,        # true rewards - should be noisy. use for evaluation.
            "expert_rewards": noiseless_rewards, # expert rewards - should be noiseless (else expert is anticipating noise). 
                                                 # use for expert decision-making
        })

    # # Visualize true vs expert rewards (should see less noise in the latter)
    # # for action 0.
    # import matplotlib.pyplot as plt
    # action_to_viz = 0
    # plt.plot([round["true_rewards"][action_to_viz] for round in data["rounds"]], label="True rewards")
    # plt.plot([round["expert_rewards"][action_to_viz] for round in data["rounds"]], label="Expert rewards")
    # plt.legend()
    # plt.savefig(f"fig_{noise_std}.png")
    # plt.close()
    return data

def main(args):
    # Generate the data
    data = generate_data(T=args.T, n_actions=2, n_features=2, noise_std=args.noise_std, seed=args.seed)

    # Save data to a pickle file
    with open(args.output_file, 'wb') as f:
        pkl.dump(data, f)
    
    print(f"Data for {args.T} rounds generated and saved to {args.output_file} with seed {args.seed}")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

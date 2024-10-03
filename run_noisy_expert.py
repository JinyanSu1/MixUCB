import numpy as np
import pickle
import argparse
from tqdm import tqdm
import logging
import torch

logging.basicConfig(filename='simulation_NoisyExpert.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')



def run_NoisyExpert(data, T, temperature):
    CR_NoisyExpert = []
    r_NoisyExpert = 0

    # Iterate over the rounds stored in the data
    for i in tqdm(range(T)):
        logging.info(f'Running NoisyExpert - round: {i}')
        true_rewards = data["rounds"][i]["true_rewards"]

        # Sample expert action using softmax based on true rewards and temperature
        rewards_tensor = torch.tensor(true_rewards, dtype=torch.float32)

        # Use torch.softmax for temperature scaling
        action_probs = torch.softmax(rewards_tensor * temperature, dim=0).numpy()

        # Sample expert action based on probabilities
        noisy_action = np.random.choice(len(true_rewards), p=action_probs)

        # Get the reward for the noisy expert action
        reward = true_rewards[noisy_action]
        
        # Update cumulative reward
        r_NoisyExpert += reward
        CR_NoisyExpert.append(r_NoisyExpert)

        logging.info(f'NoisyExpert: {r_NoisyExpert}')

    return CR_NoisyExpert

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run NoisyExpert Baseline with pre-generated data from a pickle file')
    parser.add_argument('--T', type=int, default=1000, help='Number of rounds to run')
    parser.add_argument('--temperature', type=float, default=70, help='Temperature for softmax expert action sampling')
    parser.add_argument('--pickle_file', type=str, default = 'simulation_data.pkl', help='Path to the pickle file containing pre-generated data')
    
    args = parser.parse_args()

    # Load pre-generated data from the pickle file
    with open(args.pickle_file, 'rb') as f:
        data = pickle.load(f)
    
    # Extract the number of rounds (T) from the data
    T = args.T if args.T <= len(data["rounds"]) else len(data["rounds"])

    # Run NoisyExpert using the pre-generated data
    CR_NoisyExpert = run_NoisyExpert(data, T, args.temperature)

    print(f"Finished running NoisyExpert for {T} rounds.")

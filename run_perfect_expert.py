import numpy as np
import pickle
import argparse
from tqdm import tqdm
import logging
import os
import time

logging.basicConfig(filename='simulation_PerfectExpert.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def run_PerfectExpert(data, T):
    CR_PerfectExpert = []
    r_PerfectExpert = 0

    # Iterate over the rounds stored in the data
    for i in tqdm(range(T)):
        logging.info(f'Running PerfectExpert - round: {i}')
        true_rewards = data["rounds"][i]["true_rewards"]

        # Select the action with the highest true reward (perfect expert action)
        best_action = np.argmax(true_rewards)
        reward = true_rewards[best_action]
        
        # Update cumulative reward
        r_PerfectExpert += reward
        CR_PerfectExpert.append(r_PerfectExpert)

        logging.info(f'PerfectExpert: {r_PerfectExpert}')

    return CR_PerfectExpert

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run PerfectExpert Baseline with pre-generated data from a pickle file')
    parser.add_argument('--T', type=int, default=1000, help='Number of rounds to run')
    parser.add_argument('--pickle_file', type=str, default = 'simulation_data.pkl', help='Path to the pickle file containing pre-generated data')
    
    args = parser.parse_args()

    # Load pre-generated data from the pickle file
    with open(args.pickle_file, 'rb') as f:
        data = pickle.load(f)
    
    # Extract the number of rounds (T) from the data
    T = args.T if args.T <= len(data["rounds"]) else len(data["rounds"])

    # Run PerfectExpert using the pre-generated data
    CR_PerfectExpert = run_PerfectExpert(data, T)

    print(f"Finished running PerfectExpert for {T} rounds.")

    results = 'perfect_expert_results'
    os.makedirs(results, exist_ok=True)
    pkl_name = os.path.join(results, f'{time.strftime("%Y%m%d_%H%M%S")}.pkl')
    dict_to_save = {
        'CR_PerfectExpert': CR_PerfectExpert,
        'T': args.T,
    }
    with open(pkl_name, 'wb') as f:
        pickle.dump(dict_to_save, f)
    print('Saved to {}'.format(pkl_name))
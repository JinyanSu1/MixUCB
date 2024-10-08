import numpy as np
import pickle
from utils.linucb import LinUCB
import argparse
from tqdm import tqdm
import logging
import os
import time

logging.basicConfig(filename='simulation_linucb.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

parser = argparse.ArgumentParser(description='Run LinUCB Baseline with pre-generated data from a pickle file')
parser.add_argument('--T', type=int, default=1000)
parser.add_argument('--lambda_', type=float, default=0.001)
parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('--pickle_file', type=str, default = 'simulation_data.pkl', help='Path to the pickle file containing pre-generated data')
parser.add_argument('--setting_id', type=int, default=0)

def run_linucb(data, T, linucb):
    CR_linucb = []
    r_linucb = 0

    # Iterate over the rounds stored in the data
    for i in tqdm(range(T)):
        logging.info(f'Running LinUCB - round: {i}')
        
        # Load pre-generated context and rewards for the current round
        context = data["rounds"][i]["context"]
        true_rewards = data["rounds"][i]["true_rewards"]

        # Select action and calculate reward
        linucb_ucb, _ = linucb.get_ucb_lcb(context)
        action_hat = np.argmax(linucb_ucb)
        reward = true_rewards[action_hat]

        # Update LinUCB with the selected action
        linucb.update(action_hat, context, reward)

        # Track cumulative reward
        r_linucb += reward
        CR_linucb.append(r_linucb)

        logging.info(f'LinUCB: {r_linucb}')

    return CR_linucb

def main(args):
    # Load pre-generated data from the pickle file
    with open(args.pickle_file, 'rb') as f:
        data = pickle.load(f)
    
    # Extract the number of rounds (T) from the data
    T = args.T if args.T <= len(data["rounds"]) else len(data["rounds"])
    
    n_actions = len(data["rounds"][0]["true_rewards"])  # Number of actions
    print(f"Number of actions: {n_actions}")
    n_features = data["rounds"][0]["context"].shape[1]
    print(f"Number of features: {n_features}")
    alpha = args.alpha

    # Initialize LinUCB
    linucb = LinUCB(n_actions, n_features, alpha, args.lambda_)

    # Run LinUCB using the pre-generated data
    CR_linucb = run_linucb(data, T, linucb)

    print(f"Finished running LinUCB for {T} rounds.")

    results = f'linucb_results_{args.setting_id}'
    os.makedirs(results, exist_ok=True)
    pkl_name = os.path.join(results, f'{time.strftime("%Y%m%d_%H%M%S")}.pkl')
    dict_to_save = {
        'CR_linucb': CR_linucb,
        'alpha': args.alpha,
        'lambda_': args.lambda_,
        'T': args.T,
        'n_actions': n_actions,
        'n_features': n_features,
    }
    with open(pkl_name, 'wb') as f:
        pickle.dump(dict_to_save, f)
    print('Saved to {}'.format(pkl_name))

if __name__ == "__main__":    
    args = parser.parse_args()

    main(args)

import numpy as np
import pickle
from utils.linucb import LinUCB
import argparse
from tqdm import tqdm
import logging
import os
import time

logging.basicConfig(filename='simulation_mixucbIII.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

parser = argparse.ArgumentParser(description='Run MixUCB-III Baseline')
parser.add_argument('--T', type=int, default=1000)
parser.add_argument('--delta', nargs='+', type=float, default=[0.2, 0.5, 1.,2., 5.])
parser.add_argument('--lambda_', type=float, default=0.001)
parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('--pickle_file', type=str, default='simulation_data.pkl', help='Path to the pickle file containing pre-generated data')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument("--setting_id", type=str, default="0", help="Setting ID for the experiment")

def run_mixucbIII(data, T, n_actions, delta, mixucbIII):
    CR_mixucbIII = []
    q_mixucbIII = np.zeros(T)
    TotalQ_mixucbIII = 0
    r_mixucbIII = 0

    for i in tqdm(range(T)):
        logging.info(f'Running MixUCB-III - round: {i}')
        
        # Load pre-generated context and rewards for the current round
        context = data["rounds"][i]["context"]
        true_rewards = data["rounds"][i]["true_rewards"]
        
        # Calculate UCB and LCB
        mixucb_ucb, mixucb_lcb = mixucbIII.get_ucb_lcb(context)
        width = mixucb_ucb - mixucb_lcb
        action_hat = np.argmax(mixucb_ucb)
        width_Ahat = width[action_hat]

        # Determine if querying expert or not
        if width_Ahat > delta:
            TotalQ_mixucbIII += 1
            expert_action = np.argmax(true_rewards)
            reward = true_rewards[expert_action]
            mixucbIII.update_all(context, true_rewards)
            q_mixucbIII[i] = 1
        else:
            reward = true_rewards[action_hat]
            mixucbIII.update(action_hat, context, reward)

        # Accumulate rewards and log results
        r_mixucbIII += reward
        CR_mixucbIII.append(r_mixucbIII)

        logging.info(f'MixUCB-III: {r_mixucbIII}, TotalQ_mixucbIII: {TotalQ_mixucbIII}, q_mixucbIII: {q_mixucbIII[i]}')

    return CR_mixucbIII, TotalQ_mixucbIII, q_mixucbIII

def main(args):
    # Set random seed for reproducibility
    np.random.seed(args.seed)

    # Load pre-generated data from the pickle file
    with open(args.pickle_file, 'rb') as f:
        data = pickle.load(f)

    # Extract n_actions and n_features from the data
    n_actions = len(data["rounds"][0]["true_rewards"])  # Number of actions
    n_features = data["rounds"][0]["context"].shape[1]
    # Extract the number of rounds (T) from the data
    T = args.T if args.T <= len(data["rounds"]) else len(data["rounds"])

    # Initialize parameters
    delta_list = args.delta
    alpha = args.alpha
    setting_id = args.setting_id

    for delta in delta_list:
        results = os.path.join(f'mixucbIII_results_{setting_id}', '{}'.format(delta))
        os.makedirs(results, exist_ok=True)
        print('Makedir {}'.format(results))
        for rep_id in range(5):
            # Initialize MixUCB-III model
            mixucbIII = LinUCB(n_actions, n_features, alpha, args.lambda_)

            # Run MixUCB-III using the pre-generated data
            CR_mixucbIII, TotalQ_mixucbIII, q_mixucbIII = run_mixucbIII(data, T, n_actions, delta, mixucbIII)

            print(f"Finished running MixUCB-III for {T} rounds.")

            pkl_name = os.path.join(results, f'{time.strftime("%Y%m%d_%H%M%S")}.pkl')
            dict_to_save = {
                'CR_mixucbIII': CR_mixucbIII,
                'alpha': args.alpha,
                'lambda_': args.lambda_,
                'T': args.T,
                'n_actions': n_actions,
                'n_features': n_features,
                'delta': delta,
                'TotalQ_mixucbIII': TotalQ_mixucbIII,
                'q_mixucbIII': q_mixucbIII,
            }
            with open(pkl_name, 'wb') as f:
                pickle.dump(dict_to_save, f)
            print('Saved to {}'.format(pkl_name))

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
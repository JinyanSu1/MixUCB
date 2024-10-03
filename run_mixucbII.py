import numpy as np
import torch
import pickle
from utils.run_simulation import CBOptimizationDPP
from utils.linucb import LinUCB, OnlineLogisticRegressionOracle
import argparse
from tqdm import tqdm
import logging
from scipy.linalg import inv, sqrtm
import os
import time

logging.basicConfig(filename='simulation_mixucbII.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def softmax_with_temperature(rewards, temperature):
    """Compute the softmax of rewards scaled by temperature."""
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
    action_probs = torch.softmax(rewards_tensor * temperature, dim=0).numpy()
    return action_probs

def run_mixucbII(data, T, n_actions, delta, temperature, mixucbII_query_part, mixucbII_NotQuery_part):
    CR_mixucbII = []
    q_mixucbII = np.zeros(T)
    TotalQ_mixucbII = 0
    r_mixucbII = 0

    ### FOR DPP
    opt_probDPP = CBOptimizationDPP(n_actions, mixucbII_NotQuery_part.n_features, mixucbII_NotQuery_part.alpha**2, mixucbII_query_part.beta)
    ### END
    for i in tqdm(range(T)):
        logging.info(f'Running MixUCB-II - round: {i}')
        
        # Load pre-generated context and rewards for the current round
        context = data["rounds"][i]["context"]
        true_rewards = data["rounds"][i]["true_rewards"]

        # Use softmax with temperature to get noisy expert action
        action_probs = softmax_with_temperature(true_rewards, temperature)
        noisy_expert_action = np.random.choice(n_actions, p=action_probs)

        actions_ucb = np.zeros(n_actions)
        
        ### FOR DPP
        As = [mixucbII_NotQuery_part.A[a] for a in range(n_actions)]
        theta_sq = mixucbII_NotQuery_part.get_theta()
        theta_lr, X_sum = mixucbII_query_part.get_optimization_parameters()
        As_sqrt = [sqrtm(A) for A in As]
        X_sum_sqrt = sqrtm(X_sum)


        actions_ucb = opt_probDPP.solve_allactions(context.flatten(), np.array(theta_sq), theta_lr, 
                                                   As, As_sqrt, X_sum, X_sum_sqrt, multithreading=False)
        action_hat = np.argmax(actions_ucb)
        
        action_hat_lcb = opt_probDPP.solve(context.flatten(), np.array(theta_sq), theta_lr, As, As_sqrt, X_sum, X_sum_sqrt, action_hat, ucb=False)

        ### ADDITION
        
        width_Ahat = actions_ucb[action_hat] - action_hat_lcb

        if width_Ahat > delta:
            TotalQ_mixucbII += 1
            mixucbII_query_part.update(context, noisy_expert_action)
            q_mixucbII[i] = 1
            reward = true_rewards[noisy_expert_action]
            mixucbII_NotQuery_part.update(noisy_expert_action, context, reward)
        else:
            reward = true_rewards[action_hat]
            mixucbII_NotQuery_part.update(action_hat, context, reward)

        r_mixucbII += reward
        CR_mixucbII.append(r_mixucbII)

        logging.info(f'MixUCB-II: {r_mixucbII}, TotalQ_mixucbII: {TotalQ_mixucbII}, q_mixucbII: {q_mixucbII[i]}')

    return CR_mixucbII, TotalQ_mixucbII, q_mixucbII

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run MixUCB-II Baseline with pre-generated data from a pickle file')
    parser.add_argument('--T', type=int, default=1000)
    parser.add_argument('--delta', nargs='+', type=float, default=[0.2, 0.5, 1.,2., 5.])
    parser.add_argument('--lambda_', type=float, default=0.001)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=100)
    parser.add_argument('--beta_MixUCBII', type=float, default=3000)
    parser.add_argument('--temperature', type=float, default=50)
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--pickle_file', type=str, default='simulation_data.pkl', help='Path to the pickle file containing pre-generated data')
    
    args = parser.parse_args()

    # Set random seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load pre-generated data from the pickle file
    with open(args.pickle_file, 'rb') as f:
        data = pickle.load(f)
    
    # Extract the number of rounds (T) from the data
    T = args.T if args.T <= len(data["rounds"]) else len(data["rounds"])
    
    n_actions = len(data["rounds"][0]["true_rewards"])  # Number of actions
    n_features = data["rounds"][0]["context"].shape[1]
    delta_list = args.delta
    beta = args.beta_MixUCBII
    temperature = args.temperature
    lambda_ = args.lambda_
    learning_rate = args.learning_rate
    alpha = args.alpha

    for delta in delta_list:
        results = os.path.join('mixucbII_results', '{}'.format(delta))
        os.makedirs(results, exist_ok=True)
        print('Makedir {}'.format(results))
        for rep_id in range(5):
            # Initialize query and non-query parts
            mixucbII_query_part = OnlineLogisticRegressionOracle(n_features, n_actions, learning_rate, lambda_, beta)
            mixucbII_NotQuery_part = LinUCB(n_actions, n_features, alpha, lambda_)

            # Run MixUCB-II using the pre-generated data
            CR_mixucbII, TotalQ_mixucbII, q_mixucbII = run_mixucbII(data, T, n_actions, delta, temperature, mixucbII_query_part, mixucbII_NotQuery_part)

            print(f"Finished running MixUCB-II for {T} rounds.")

            pkl_name = os.path.join(results, f'{time.strftime("%Y%m%d_%H%M%S")}.pkl')
            dict_to_save = {
                'CR_mixucbII': CR_mixucbII,
                'alpha': args.alpha,
                'lambda_': args.lambda_,
                'T': args.T,
                'n_actions': n_actions,
                'n_features': n_features,
                'delta': delta,
                'beta': beta,
                'temperature': temperature,
                'lr': learning_rate,
                'TotalQ_mixucbII': TotalQ_mixucbII,
                'q_mixucbII': q_mixucbII,
            }
            with open(pkl_name, 'wb') as f:
                pickle.dump(dict_to_save, f)
            print('Saved to {}'.format(pkl_name))
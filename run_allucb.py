#!/usr/bin/env python3
import numpy as np
import pickle
from utils.regression_ucb import MixUCB, OnlineLogisticRegressionOracle
import argparse
from tqdm import tqdm
import logging
import os
import time
from icecream import ic
import argparse

logging.basicConfig(filename='simulation.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def run_mixucb(data, T, n_actions, delta, online_reg_oracle, mode=None):
    assert mode in ['lin','mixI','mixII','mixIII']
    rationality = 1 # TODO make argument
    reward_per_time = np.zeros(T)
    query_per_time = np.zeros(T)
    action_per_time = []

    data_len = len(data["rounds"])
    permutation = np.arange(data_len, dtype=int)

    for i in tqdm(range(T)):
        logging.info(f'Running UCB {mode} - round: {i}')

        # Compute data index if i > T
        # (NOTE: np.random.shuffle is already seeded in the main block.)
        if i == 0:
            data_ind = i
        else:
            # if the data length is less than T, we loop over a randomly shuffled permutation of the data.
            ind = i % data_len
            if ind == 0:
                np.random.shuffle(permutation) # TODO random seed
            data_ind = permutation[ind]
        
        # Load pre-generated context and rewards for the current round
        context = data["rounds"][data_ind]["context"]
        # expected vs actual rewards
        expected_rewards = data["rounds"][data_ind]["expected_rewards"] # empty for classification datasets
        actual_rewards = data["rounds"][data_ind]["actual_rewards"]
        # noisy expert choice.
        noisy_expert_choice = data["rounds"][data_ind]["noisy_expert_choice"]
        
        # Calculate UCB and LCB
        ucb,lcb = online_reg_oracle.get_ucb_lcb(context)
        action_hat = np.argmax(ucb)

        if mode in ['mixI', 'mixII', 'mixIII']:
            # Determine if querying expert or not
            width = ucb - lcb
            width_Ahat = width[action_hat]
            # Print i, action_hat, width_Ahat, delta.
            logging.info(f'Round {i}, action_hat: {action_hat}, width_Ahat: {width_Ahat}, delta: {delta}')
            if width_Ahat > delta:
                # query expert and update online regression
                query_per_time[i] = 1

                if mode == 'mixIII':
                    # todo: should mixIII expert know expected or noisy rewards?
                    if len(expected_rewards) == 0:
                        expert_action = np.argmax(actual_rewards)
                    else:
                        expert_action = np.argmax(expected_rewards)
                else:
                    if len(expected_rewards) == 0:
                        expert_action = np.argmax(actual_rewards)
                    else:
                        expert_action = noisy_expert_choice
                reward = actual_rewards[expert_action]
                
                if mode == 'mixIII':
                    # todo: should mixIII expert know expected or noisy rewards?
                    if len(expected_rewards) == 0:
                        online_reg_oracle.update(context,rewards=actual_rewards)
                    else:
                        online_reg_oracle.update(context,rewards=expected_rewards)
                elif mode == 'mixII':
                    online_reg_oracle.update(context, action=expert_action) 
                    online_reg_oracle.update(context, action=expert_action, reward=reward)
                elif mode == 'mixI':
                    online_reg_oracle.update(context, action=expert_action)
            else:
                # take action_hat and update online regression
                reward = actual_rewards[action_hat]
                online_reg_oracle.update(context, action=action_hat, reward=reward)

        else:
            # Print i, action_hat, width_Ahat, delta.
            logging.info(f'Round {i}, action_hat: {action_hat}')
            reward = actual_rewards[action_hat]
            online_reg_oracle.update(context, action=action_hat, reward=reward)

        reward_per_time[i] = reward

        if query_per_time[i]:
            action_per_time.append(expert_action)
        else:
            action_per_time.append(action_hat)

        logging.info(f'{mode} UCB: reward {reward}, query: {query_per_time[i]}, totalq: {np.sum(query_per_time)}')

    return reward_per_time, query_per_time, action_per_time

def run_linear_oracle(data, T, theta):
    reward_per_time = np.zeros(T)
    action_per_time = []


    data_len = len(data["rounds"])
    permutation = np.arange(data_len, dtype=int)

    for i in tqdm(range(T)):
        logging.info(f'Running linear oracle - round: {i}')

        # Compute data index if i > T
        # (NOTE: np.random.shuffle is already seeded in the main block.)
        if i == 0:
            data_ind = i
        else:
            ind = i % data_len
            if ind == 0:
                np.random.shuffle(permutation)
            data_ind = permutation[ind]
        
        # Load pre-generated context and rewards for the current round
        context = data["rounds"][data_ind]["context"]

        actual_rewards = data["rounds"][data_ind]["actual_rewards"]

        est_rewards = np.dot(theta, context.ravel())
        action = np.argmax(est_rewards)

        reward = actual_rewards[action]
        reward_per_time[i] = reward
        action_per_time.append(action)

        logging.info(f'oracle: reward {reward}')

    return reward_per_time, action_per_time

    pass

def run_expert(data, T, type='perfect_exp'):
    """
    Runs the expert algorithm for T rounds (either perfect or noisy).
    """
    reward_per_time = np.zeros(T)
    action_per_time = []

    data_len = len(data["rounds"])
    permutation = np.arange(data_len, dtype=int)

    for i in tqdm(range(T)):
        logging.info(f'Running expert type {type} - round: {i}')

        # Compute data index if i > T
        # (NOTE: np.random.shuffle is already seeded in the main block.)
        if i == 0:
            data_ind = i
        else:
            ind = i % data_len
            if ind == 0:
                np.random.shuffle(permutation)
            data_ind = permutation[ind]
        
        # Load pre-generated expected and actual rewards for the current round
        expected_rewards = data["rounds"][data_ind]["expected_rewards"]
        actual_rewards = data["rounds"][data_ind]["actual_rewards"]
        # noisy expert choice.
        noisy_expert_choice = data["rounds"][data_ind]["noisy_expert_choice"]

        # Select either the perfect or noisy expert action based on the type.
        # Identical to logic in run_mixucb for MixUCB-I/II/III expert logic.
        if type == 'perfect_exp':
            # Same as MixUCB-III expert logic
            if len(expected_rewards) == 0:
                expert_action = np.argmax(actual_rewards)
            else:
                expert_action = np.argmax(expected_rewards)
        elif type == 'noisy_exp':
            # Same as MixUCB-I/II expert logic.
            if len(expected_rewards) == 0:
                expert_action = np.argmax(actual_rewards)
            else:
                expert_action = noisy_expert_choice
        
        reward = actual_rewards[expert_action]
        reward_per_time[i] = reward
        action_per_time.append(expert_action)

        logging.info(f'expert: reward {reward}')

    return reward_per_time, action_per_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run UCB Baseline + oracles + experts.')
    parser.add_argument('--mode', type=str, default='lin', help='must be: lin, mixI, mixII, mixIII, sq_oracle, lr_oracle, perfect_exp, noisy_exp')
    parser.add_argument('--T', type=int, default=0)
    parser.add_argument('--delta', nargs='+', type=float, default=[4., 5., 6., 7., 8.])
    parser.add_argument('--lambda_', type=float, default=0.001, help='regularization weight')
    parser.add_argument('--pickle_file', type=str, default='simulation_data.pkl', help='Path to the pickle file containing pre-generated data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument("--data_name", type=str, default='', help="Name of the dataset to use.")
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=100, help='squared loss confidence radius')
    parser.add_argument('--beta', type=float, default=1000, help='log loss confidence radius')
    # TODO add temperature?
    
    args = parser.parse_args()

    # Set random seed for reproducibility
    np.random.seed(args.seed)

    # Load pre-generated data from the pickle file
    with open(args.pickle_file, 'rb') as f:
        data = pickle.load(f)

    # Extract n_actions and n_features from the data
    n_actions = len(data["rounds"][0]["actual_rewards"])
    n_features = data["rounds"][0]["context"].shape[1]
    # Extract the number of rounds (T) from the data
    T = args.T if args.T > 0 else len(data["rounds"])

    # Initialize parameters
    delta_list = args.delta
    alpha = args.alpha
    beta = args.beta
    lambda_ = args.lambda_ # TODO two diff lambda
    learning_rate = args.learning_rate
    data_name = args.data_name
    mode = args.mode
    seed = args.seed

    if mode in ['sq_oracle', 'lr_oracle']:
        results = os.path.join(data_name, f"seed_{seed:02d}", '{}_results'.format(mode))
        os.makedirs(results, exist_ok=True)
        print('Makedir {}'.format(results))

        if mode == 'sq_oracle':
            theta = data['true_theta']
        else:
            theta = data['true_theta_classification']

        # Run using the pre-generated data
        reward_per_time, action_per_time = run_linear_oracle(data, T, theta)

        print(f"Finished running {mode} for {T} rounds.")

        pkl_name = os.path.join(results, f'{time.strftime("%Y%m%d_%H%M%S")}.pkl')
        dict_to_save = {
            'reward_per_time': reward_per_time,
            'action_per_time': action_per_time,
            'delta_list': delta_list
        }
        with open(pkl_name, 'wb') as f:
            pickle.dump(dict_to_save, f)
        print('Saved to {}'.format(pkl_name))
    elif mode in ['perfect_exp', 'noisy_exp']:
        results = os.path.join(data_name, f"seed_{seed:02d}", '{}_results'.format(mode))
        os.makedirs(results, exist_ok=True)
        print('Makedir {}'.format(results))
        
        reward_per_time, action_per_time = run_expert(data, T, type=mode)
        
        print(f"Finished running {mode} for {T} rounds.")

        pkl_name = os.path.join(results, f'{time.strftime("%Y%m%d_%H%M%S")}.pkl')
        dict_to_save = {
            'reward_per_time': reward_per_time,
            'action_per_time': action_per_time,
            'delta_list': delta_list
        }
        with open(pkl_name, 'wb') as f:
            pickle.dump(dict_to_save, f)
        print('Saved to {}'.format(pkl_name))

    else:
        if mode == 'lin':
            delta_list = [0]
        
        for delta in delta_list:
            results = os.path.join(data_name, f"seed_{seed:02d}", '{}_ucb_results'.format(mode), '{}'.format(delta))
            os.makedirs(results, exist_ok=True)
            print('Makedir {}'.format(results))

            online_reg_oracle = OnlineLogisticRegressionOracle(n_features, n_actions, learning_rate, lambda_, beta, rad_sq=alpha)

            # Run using the pre-generated data
            reward_per_time, query_per_time, action_per_time = run_mixucb(data, T, n_actions, delta, online_reg_oracle, mode=mode)

            print(f"Finished running {mode} UCB for {T} rounds.")

            pkl_name = os.path.join(results, f'{time.strftime("%Y%m%d_%H%M%S")}.pkl')
            dict_to_save = {
                'reward_per_time': reward_per_time,
                'query_per_time': query_per_time,
                'action_per_time': action_per_time,
                'alpha': args.alpha,
                'beta': args.beta,
                'lambda_': args.lambda_,
                'delta_list': delta_list
            }
            with open(pkl_name, 'wb') as f:
                pickle.dump(dict_to_save, f)
            print('Saved to {}'.format(pkl_name))

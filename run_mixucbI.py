import numpy as np
import pickle
from utils.run_simulation import solve_convex_optimization_ucb, solve_convex_optimization_lcb
from utils.linucb import LinUCB, OnlineLogisticRegressionOracle
import argparse
from tqdm import tqdm
import logging

logging.basicConfig(filename='simulation_mixucbI.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def run_mixucbI(data, T, n_actions, delta, mixucbI_query_part, mixucbI_NotQuery_part):
    CR_mixucbI = []
    q_mixucbI = np.zeros(T)
    TotalQ_mixucbI = 0
    r_mixucbI = 0

    # Iterate over the rounds stored in the data
    for i in tqdm(range(T)):
        logging.info(f'Running MixUCB-I - round: {i}')
        
        # Load pre-generated context and rewards for the current round
        context = data["rounds"][i]["context"]
        true_rewards = data["rounds"][i]["true_rewards"]
        expert_action = np.argmax(true_rewards)  # Expert action is the one with max reward

        actions_ucb = np.zeros(n_actions)
        for obj_a in range(n_actions):
            actions_ucb[obj_a] = solve_convex_optimization_ucb(obj_a, context, mixucbI_query_part, mixucbI_NotQuery_part, n_actions)

        action_hat = np.argmax(actions_ucb)
        action_hat_lcb = solve_convex_optimization_lcb(action_hat, context, mixucbI_query_part, mixucbI_NotQuery_part, n_actions)
        width_Ahat = actions_ucb[action_hat] - action_hat_lcb

        if width_Ahat > delta:
            TotalQ_mixucbI += 1
            mixucbI_query_part.update(context, expert_action)
            q_mixucbI[i] = 1
            reward = true_rewards[expert_action]
        else:
            reward = true_rewards[action_hat]
            mixucbI_NotQuery_part.update(action_hat, context, reward)

        r_mixucbI += reward
        CR_mixucbI.append(r_mixucbI)

        logging.info(f'MixUCB-I: {r_mixucbI}, TotalQ_mixucbI: {TotalQ_mixucbI}, q_mixucbI: {q_mixucbI[i]}')

    return CR_mixucbI, TotalQ_mixucbI, q_mixucbI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run MixUCB-I Baseline with pre-generated data from a pickle file')
    parser.add_argument('--T', type=int, default=1000)
    parser.add_argument('--delta', type=float, default=0.2)
    parser.add_argument('--lambda_', type=float, default=0.001)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--n_actions', type=int, default=30)
    parser.add_argument('--n_features', type=int, default=2)
    parser.add_argument('--beta', type=float, default=3.0)
    parser.add_argument('--temperature', type=float, default=0.01)
    parser.add_argument('--pickle_file', type=str, default='simulation_data.pkl', help='Path to the pickle file containing pre-generated data')
    
    args = parser.parse_args()

    # Load pre-generated data from the pickle file
    with open(args.pickle_file, 'rb') as f:
        data = pickle.load(f)
    
    # Extract the number of rounds (T) from the data
    T = args.T if args.T <= len(data["rounds"]) else len(data["rounds"])
    
    n_actions = args.n_actions
    n_features = args.n_features
    delta = args.delta
    lambda_ = args.lambda_
    learning_rate = args.learning_rate
    alpha = args.alpha

    # Initialize query and non-query parts
    Query_part = OnlineLogisticRegressionOracle(n_features, n_actions, learning_rate, lambda_)
    NotQuery_part = LinUCB(n_actions, n_features, alpha, lambda_)

    # Run MixUCB-I using the pre-generated data
    CR_mixucbI, TotalQ_mixucbI, q_mixucbI = run_mixucbI(data, T, n_actions, delta, Query_part, NotQuery_part)

    print(f"Finished running MixUCB-I for {T} rounds.")

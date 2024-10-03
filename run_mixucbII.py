import numpy as np
from utils.run_simulation import solve_convex_optimization_ucb, solve_convex_optimization_lcb
from utils.linucb import LinUCB, OnlineLogisticRegressionOracle
from utils.get_data import ContextGenerator
import argparse
from tqdm import tqdm
import logging

logging.basicConfig(filename='simulation_mixucbII.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def run_mixucbII(generator, T, n_actions, delta, mixucbII_query_part, mixucbII_NotQuery_part):
    CR_mixucbII = []
    q_mixucbII = np.zeros(T)
    TotalQ_mixucbII = 0
    r_mixucbII = 0

    for i in tqdm(range(T)):
        logging.info(f'Running MixUCB-II - round: {i}')
        context, true_rewards, expert_action = generator.generate_context_rewards_and_expert_action()

        actions_ucb = np.zeros(n_actions)
        for obj_a in range(n_actions):
            actions_ucb[obj_a] = solve_convex_optimization_ucb(obj_a, context, mixucbII_query_part, mixucbII_NotQuery_part, n_actions)

        action_hat = np.argmax(actions_ucb)
        action_hat_lcb = solve_convex_optimization_lcb(action_hat, context, mixucbII_query_part, mixucbII_NotQuery_part, n_actions)
        width_Ahat = actions_ucb[action_hat] - action_hat_lcb

        if width_Ahat > delta:
            TotalQ_mixucbII += 1
            mixucbII_query_part.update(context, expert_action)
            q_mixucbII[i] = 1
            reward = true_rewards[expert_action]
            mixucbII_NotQuery_part.update(expert_action, context, reward)
        else:
            reward = true_rewards[action_hat]
            mixucbII_NotQuery_part.update(action_hat, context, reward)

        r_mixucbII += reward
        CR_mixucbII.append(r_mixucbII)

        logging.info(f'MixUCB-II: {r_mixucbII}, TotalQ_mixucbII: {TotalQ_mixucbII}, q_mixucbII: {q_mixucbII[i]}')

    return CR_mixucbII, TotalQ_mixucbII, q_mixucbII

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run MixUCB-II Baseline')
    parser.add_argument('--T', type=int, default=1000)
    parser.add_argument('--delta', type=float, default=0.2)
    parser.add_argument('--lambda_', type=float, default=0.001)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--n_actions', type=int, default=30)
    parser.add_argument('--n_features', type=int, default=2)
    parser.add_argument('--beta', type=float, default=3.0)
    parser.add_argument('--temperature', type=float, default=0.01)
    args = parser.parse_args()

    np.random.seed(42)
    T = args.T
    delta = args.delta
    n_actions = args.n_actions
    n_features = args.n_features
    lambda_ = args.lambda_
    learning_rate = args.learning_rate
    alpha = args.alpha
    beta = args.beta
    temperature = args.temperature

    true_weights = np.random.randn(n_actions, n_features)
    norm = np.linalg.norm(true_weights)
    if norm > 1:
        true_weights = true_weights / norm
    generator = ContextGenerator(true_weights=true_weights, noise_std=0, temperature=temperature)

    Query_part = OnlineLogisticRegressionOracle(n_features, n_actions, learning_rate, lambda_, beta)
    NotQuery_part = LinUCB(n_actions, n_features, alpha, lambda_)

    CR_mixucbII, TotalQ_mixucbII, q_mixucbII = run_mixucbII(generator, T, n_actions, delta, Query_part, NotQuery_part)

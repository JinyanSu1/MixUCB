import numpy as np
from utils.linucb import LinUCB
from utils.get_data import ContextGenerator
import argparse
from tqdm import tqdm
import logging

logging.basicConfig(filename='simulation_mixucbIII.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def run_mixucbIII(generator, T, n_actions, delta, mixucbIII):
    CR_mixucbIII = []
    q_mixucbIII = np.zeros(T)
    TotalQ_mixucbIII = 0
    r_mixucbIII = 0

    for i in tqdm(range(T)):
        logging.info(f'Running MixUCB-III - round: {i}')
        context, true_rewards, expert_action = generator.generate_context_rewards_and_expert_action()

        mixucb_ucb, mixucb_lcb = mixucbIII.get_ucb_lcb(context)
        width = mixucb_ucb - mixucb_lcb
        action_hat = np.argmax(mixucb_ucb)
        width_Ahat = width[action_hat]

        if width_Ahat > delta:
            TotalQ_mixucbIII += 1
            expert_action = np.argmax(true_rewards)
            reward = true_rewards[expert_action]
            mixucbIII.update_all(context, true_rewards)
            q_mixucbIII[i] = 1
        else:
            reward = true_rewards[action_hat]
            mixucbIII.update(action_hat, context, reward)

        r_mixucbIII += reward
        CR_mixucbIII.append(r_mixucbIII)

        logging.info(f'MixUCB-III: {r_mixucbIII}, TotalQ_mixucbIII: {TotalQ_mixucbIII}, q_mixucbIII: {q_mixucbIII[i]}')

    return CR_mixucbIII, TotalQ_mixucbIII, q_mixucbIII

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run MixUCB-III Baseline')
    parser.add_argument('--T', type=int, default=1000)
    parser.add_argument('--delta', type=float, default=0.2)
    parser.add_argument('--lambda_', type=float, default=0.001)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--n_actions', type=int, default=30)
    parser.add_argument('--n_features', type=int, default=2)
    parser.add_argument('--temperature', type=float, default=0.01)
    args = parser.parse_args()

    np.random.seed(42)
    T = args.T
    delta = args.delta
    n_actions = args.n_actions
    n_features = args.n_features
    alpha = args.alpha
    temperature = args.temperature

    true_weights = np.random.randn(n_actions, n_features)
    norm = np.linalg.norm(true_weights)
    if norm > 1:
        true_weights = true_weights / norm
    generator = ContextGenerator(true_weights=true_weights, noise_std=0, temperature=temperature)

    mixucbIII = LinUCB(n_actions, n_features, alpha, args.lambda_)

    CR_mixucbIII, TotalQ_mixucbIII, q_mixucbIII = run_mixucbIII(generator, T, n_actions, delta, mixucbIII)

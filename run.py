import numpy as np
from utils.run_simulation import solve_convex_optimization_ucb, solve_convex_optimization_lcb
from utils.linucb import LinUCB, OnlineLogisticRegressionOracle
from utils.get_data import ContextGenerator
import argparse
from tqdm import tqdm
import logging
logging.basicConfig(filename='simulation.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
np.random.seed(42)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    
    
    parser.add_argument('--beta_MixUCBI', type=float, default=3.0)
    parser.add_argument('--beta_MixUCBII', type=float, default=3.0)
    parser.add_argument('--delta', type=float, default=0.2)
    
    
    

    parser.add_argument('--temperature', type=float, default=0.01)
    parser.add_argument('--T', type=int, default=1000)
    parser.add_argument('--lambda_', type=float, default=0.001)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--n_actions', type=int, default=30)
    parser.add_argument('--n_features', type=int, default=2)
    args = parser.parse_args()

    beta_MixUCBI = args.beta_MixUCBI
    beta_MixUCBII = args.beta_MixUCBII
    temperature = args.temperature
    n_actions = args.n_actions
    noise_std = 0
    alpha = args.alpha
    delta = args.delta
    lambda_ = args.lambda_
    learning_rate = args.learning_rate
    T = args.T
    n_features = args.n_features
    true_weights = np.random.randn(n_actions, n_features)
    norm = np.linalg.norm(true_weights)
    if norm > 1:  
        true_weights = true_weights / norm
    generator = ContextGenerator(true_weights=true_weights, noise_std=noise_std, temperature=temperature)
    
    # Currently, I am using the same parameters for all the baselines, modify here if you want to use different parameters(like alpha, lambda, etc) for different baselines
    # ------------------------start------------------------
    mixucbIII = LinUCB(n_actions, n_features, alpha, lambda_)
    linucb = LinUCB(n_actions, n_features, alpha, lambda_)
    AlwaysQueryNoisyExpert = LinUCB(n_actions, n_features, alpha, lambda_)
    AlwaysQueryFullInfo = LinUCB(n_actions, n_features, alpha, lambda_)
    
    mixucbI_query_part = OnlineLogisticRegressionOracle(n_features, n_actions, learning_rate, lambda_, beta_MixUCBI)
    mixucbI_NotQuery_part= LinUCB(n_actions, n_features, alpha, lambda_)
    
    mixucbII_query_part = OnlineLogisticRegressionOracle(n_features, n_actions, learning_rate, lambda_, beta_MixUCBII)
    mixucbII_NotQuery_part= LinUCB(n_actions, n_features, alpha, lambda_)
    # -----------------------end------------------------
    
    
    
    
    
    
    # Here are the variables to store
    # ----------------------start----------------------
    CR_mixucbIII = []
    CR_linucb = []
    CR_AlwaysQueryNoisyExpert = []
    CR_AlwaysQueryFullInfo = []
    CR_mixucbI = []
    CR_mixucbII = []
    
    q_mixucbIII = np.zeros(T)
    q_mixucbI = np.zeros(T)
    q_mixucbII = np.zeros(T)

    
    TotalQ_mixucbIII = 0
    TotalQ_mixucbI = 0
    TotalQ_mixucbII = 0
    # ----------------------end----------------------

    r_mixucbIII = 0
    r_linucb = 0
    r_AlwaysQueryNoisyExpert = 0
    r_AlwaysQueryFullInfo = 0
    r_mixucbI = 0
    r_mixucbII = 0
    
    for i in tqdm(range(T)):
        print('current round:', i)
        context, true_rewards, expert_action = generator.generate_context_rewards_and_expert_action()
        
        
        
        ## MixUCB-I
        # ----------------------start----------------------
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
        # ----------------------end----------------------



        ## MixUCB-II
        # ----------------------start----------------------
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
        # ----------------------end----------------------
        
        
        ## MixUCB-III
        # ----------------------start----------------------
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
        # ----------------------end----------------------
        
        
        
        ## LinUCB
        # ----------------------start----------------------
        linucb_ucb, _ = linucb.get_ucb_lcb(context)
        action_hat = np.argmax(linucb_ucb)
        reward = true_rewards[action_hat]
        linucb.update(action_hat, context, reward)
        r_linucb += reward
        CR_linucb.append(r_linucb)
        # ----------------------end----------------------
        
    
        ## AlwaysQueryNoisyExpert
        # ----------------------start----------------------
        reward = true_rewards[expert_action]
        r_AlwaysQueryNoisyExpert += reward
        CR_AlwaysQueryNoisyExpert.append(r_AlwaysQueryNoisyExpert)
        
        
        
        ## AlwaysQueryFullInfo
        # ----------------------start----------------------
        reward = true_rewards[np.argmax(true_rewards)]
        r_AlwaysQueryFullInfo += reward
        CR_AlwaysQueryFullInfo.append(r_AlwaysQueryFullInfo)
        # ----------------------end----------------------
        
        
        logging.info(f'MixUCB-III: {r_mixucbIII}, MixUCB-I: {r_mixucbI}, MixUCB-II: {r_mixucbII}, LinUCB: {r_linucb}, '
                     f'AlwaysQueryNoisyExpert: {r_AlwaysQueryNoisyExpert}, AlwaysQueryFullInfo: {r_AlwaysQueryFullInfo}')
        logging.info(f'TotalQ_mixucbIII: {TotalQ_mixucbIII}, TotalQ_mixucbI: {TotalQ_mixucbI}, TotalQ_mixucbII: {TotalQ_mixucbII}')
        logging.info(f'q_mixucbIII: {q_mixucbIII[i]}, q_mixucbI: {q_mixucbI[i]}, q_mixucbII: {q_mixucbII[i]}')
        
        
        
        
        





    

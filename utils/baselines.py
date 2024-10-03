import numpy as np
from run_simulation import solve_convex_optimization_ucb, solve_convex_optimization_lcb
from tqdm import tqdm
import logging
def run_mixucbI(generator, T, n_actions, delta, mixucbI_query_part, mixucbI_NotQuery_part, true_rewards):
    CR_mixucbI = []
    q_mixucbI = np.zeros(T)
    TotalQ_mixucbI = 0
    r_mixucbI = 0

    for i in tqdm(range(T)):
        logging.info(f'Running MixUCB-I - round: {i}')
        context, true_rewards, expert_action = generator.generate_context_rewards_and_expert_action()

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

def run_mixucbII(generator, T, n_actions, delta, mixucbII_query_part, mixucbII_NotQuery_part, true_rewards):
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

def run_mixucbIII(generator, T, n_actions, delta, mixucbIII, true_rewards):
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

def run_linucb(generator, T, linucb, true_rewards):
    CR_linucb = []
    r_linucb = 0

    for i in tqdm(range(T)):
        logging.info(f'Running LinUCB - round: {i}')
        context, true_rewards, expert_action = generator.generate_context_rewards_and_expert_action()

        linucb_ucb, _ = linucb.get_ucb_lcb(context)
        action_hat = np.argmax(linucb_ucb)
        reward = true_rewards[action_hat]
        linucb.update(action_hat, context, reward)

        r_linucb += reward
        CR_linucb.append(r_linucb)

        logging.info(f'LinUCB: {r_linucb}')

    return CR_linucb

def run_AlwaysQueryNoisyExpert(generator, T, true_rewards):
    CR_AlwaysQueryNoisyExpert = []
    r_AlwaysQueryNoisyExpert = 0

    for i in tqdm(range(T)):
        logging.info(f'Running AlwaysQueryNoisyExpert - round: {i}')
        context, true_rewards, expert_action = generator.generate_context_rewards_and_expert_action()

        reward = true_rewards[expert_action]
        r_AlwaysQueryNoisyExpert += reward
        CR_AlwaysQueryNoisyExpert.append(r_AlwaysQueryNoisyExpert)

        logging.info(f'AlwaysQueryNoisyExpert: {r_AlwaysQueryNoisyExpert}')

    return CR_AlwaysQueryNoisyExpert

def run_AlwaysQueryFullInfo(generator, T, true_rewards):
    CR_AlwaysQueryFullInfo = []
    r_AlwaysQueryFullInfo = 0

    for i in tqdm(range(T)):
        logging.info(f'Running AlwaysQueryFullInfo - round: {i}')
        context, true_rewards, expert_action = generator.generate_context_rewards_and_expert_action()

        reward = true_rewards[np.argmax(true_rewards)]
        r_AlwaysQueryFullInfo += reward
        CR_AlwaysQueryFullInfo.append(r_AlwaysQueryFullInfo)

        logging.info(f'AlwaysQueryFullInfo: {r_AlwaysQueryFullInfo}')

    return CR_AlwaysQueryFullInfo

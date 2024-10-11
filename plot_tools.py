import matplotlib
# matplotlib.use('TKAgg') # for g2, comment me out.
import matplotlib.pyplot as plt
import os
import pickle
import numpy as np

# fix colors for I, II, and III: orange, green, red
# fix colors for LinUCB, NoisyExpert, PerfectExpert: blue, purple, brown
color_map = {'I': 'orange', 'II': 'green', 'III': 'red', 'LinUCB': 'blue', 'NoisyExpert': 'purple', 'PerfectExpert': 'brown'}

def plot_average_rewards(axs, cumulative_rewards, cumulative_awards_std=None, params=None, ylabel='Average Reward'):
    """Plot average rewards in a 1 by m grid for different parameters."""
    for idx in range(len(axs)):
        ax = axs[idx]
        for key, item in cumulative_rewards.items():
            cumulative_reward = item[idx]
            # print('cumulative_reward', cumulative_reward)
            average_rewards = [cum_reward / (i + 1) for i, cum_reward in enumerate(cumulative_reward)]
            for elem in color_map.keys():
                if elem in key:
                    color = color_map[elem]

            if cumulative_awards_std:
                # print(key, idx)
                # print(cumulative_awards_std[key])
                std = cumulative_awards_std[key][idx]
                average_rewards_std = [s / (i + 1) for i, s in enumerate(std)]

                ax.fill_between(range(len(average_rewards)),
                                [a - s for a, s in zip(average_rewards, average_rewards_std)],
                                [a + s for a, s in zip(average_rewards, average_rewards_std)], alpha=0.2,\
                                color=color)
            if 'III' in key:
                ax.plot(average_rewards, label=f'{key}', marker='o', markersize=2, color=color)
            elif 'LinUCB' in key: # do a diamond marker
                ax.plot(average_rewards, label=f'{key}', marker='D', markersize=2, color=color)
            else:
                ax.plot(average_rewards, label=f'{key}', color=color)
            ax.set_xlabel('t')
            ax.set_ylabel(ylabel)
            if params is not None:
                ax.set_title(f'$\\Delta={params[idx]}$')
        ax.legend()

def plot_cumulative_rewards(axs, cumulative_rewards, cumulative_awards_std=None, params=None, ylabel='Cumulative Reward'):
    """Plot cumulative rewards in a 1 by m grid for different parameters."""
    for idx in range(len(axs)):
        ax = axs[idx]
        for key, item in cumulative_rewards.items():
            cumulative_reward = item[idx]
            for elem in color_map.keys():
                if elem in key:
                    color = color_map[elem]
            if cumulative_awards_std:
                std = cumulative_awards_std[key][idx]
                ax.fill_between(range(len(cumulative_reward)), [a - s for a, s in zip(cumulative_reward, std)], [a + s for a, s in zip(cumulative_reward, std)], alpha=0.2,\
                color=color)
            ax.plot(cumulative_reward, label=f'{key}',color=color)
            ax.set_xlabel('t')
            ax.set_ylabel(ylabel)
            if params is not None:
                ax.set_title(f'$\\Delta={params[idx]}$')
        ax.legend()

def plot_cumulative_queries(axs, q_mean, q_std, params):
    """Plot cumulative queries in a 1 by m grid for different parameters."""
    for idx in range(len(axs)):
        ax = axs[idx]
        for key, item in q_mean.items():
            q = item[idx]
            std = q_std[key][idx]    # putting std back in.
            for elem in color_map.keys():
                if elem in key:
                    color = color_map[elem]
            # cumsum_q = np.cumsum(q)
            ax.fill_between(range(len(q)), [a - s for a, s in zip(q, std)], [a + s for a, s in zip(q, std)], alpha=0.2,\
                color=color)
            ax.plot(q, label=f'{key}',color=color)
            ax.set_xlabel('t')
            ax.set_ylabel('Cumulative Queries')
            ax.set_title(f'$\\Delta={params[idx]}$')
        ax.legend()

def plot_six_baselines(Figure_dir='Figures',mixucb_result_postfix="",delta=0.5,result_root=''):
    os.makedirs(Figure_dir, exist_ok=True)
    CR_linucb_mean = []
    CR_linucb_std = []
    CR_mixucbI_mean = []
    CR_mixucbI_std = []
    CR_mixucbII_mean = []
    CR_mixucbII_std = []
    CR_mixucbIII_mean = []
    CR_mixucbIII_std = []
    CR_NoisyExpert_mean = []
    CR_NoisyExpert_std = []
    CR_PerfectExpert_mean = []
    CR_PerfectExpert_std = []

    # reward regret
    RR_mixucbI_mean = []
    RR_mixucbI_std = []
    RR_mixucbII_mean = []
    RR_mixucbII_std = []
    RR_mixucbIII_mean = []
    RR_mixucbIII_std = []

    # algorithm regret
    AR_mixucbI_mean = []
    AR_mixucbI_std = []
    AR_mixucbII_mean = []
    AR_mixucbII_std = []
    AR_mixucbIII_mean = []
    AR_mixucbIII_std = []

    # query_list
    q_mixUCBI_mean = []
    q_mixUCBI_std = []
    q_mixUCBII_mean = []
    q_mixUCBII_std = []
    q_mixUCBIII_mean = []
    q_mixUCBIII_std = []

    # cumulative queries (call it CQ)
    CQ_mixucbI_mean = []
    CQ_mixucbI_std = []
    CQ_mixucbII_mean = []
    CQ_mixucbII_std = []
    CQ_mixucbIII_mean = []
    CQ_mixucbIII_std = []

    perfect_expert_pkls = os.listdir(os.path.join(result_root,f'perfect_expert_results_0'))
    perfect_expert_list = []
    perfect_expert_rawreward_list = []
    for each_perfect_expert_pkl in perfect_expert_pkls:
        with open(os.path.join(result_root,'perfect_expert_results_0', each_perfect_expert_pkl), 'rb') as f:
            data = pickle.load(f)
            CR_PerfectExpert = data['CR_PerfectExpert']
            perfect_expert_list.append(CR_PerfectExpert)
            # extract raw rewards from cumulative reward list.
            raw_rewards = [CR_PerfectExpert[i] - CR_PerfectExpert[i-1] if i > 0 else CR_PerfectExpert[i] for i in range(len(CR_PerfectExpert))]
            perfect_expert_rawreward_list.append(raw_rewards)

    linucb_pkls = os.listdir(os.path.join(result_root,'linucb_results_0'))
    linucb_list = []
    for each_linucb_pkl in linucb_pkls:
        with open(os.path.join(result_root,'linucb_results_0', each_linucb_pkl), 'rb') as f:
            data = pickle.load(f)
            CR_linucb = data['CR_linucb']
            linucb_list.append(CR_linucb)

    mixucbI_pkls = os.listdir(os.path.join(result_root,f'mixucbI_results{mixucb_result_postfix}/{delta}'))
    mixucbI_list = []
    mixucbI_rawreward_list = []
    q_mixUCBI_list = []
    for each_mixucbI_pkl in mixucbI_pkls:
        with open(os.path.join(result_root,f'mixucbI_results{mixucb_result_postfix}/{delta}', each_mixucbI_pkl), 'rb') as f:
            data = pickle.load(f)
            CR_mixucbI = data['CR_mixucbI']
            mixucbI_list.append(CR_mixucbI)
            # extract raw rewards from cumulative reward list.
            raw_rewards = [CR_mixucbI[i] - CR_mixucbI[i-1] if i > 0 else CR_mixucbI[i] for i in range(len(CR_mixucbI))]
            mixucbI_rawreward_list.append(raw_rewards)
            q_mixUCBI = data['q_mixucbI']
            q_mixUCBI_list.append(q_mixUCBI)

    mixucbII_pkls = os.listdir(os.path.join(result_root,f'mixucbII_results{mixucb_result_postfix}/{delta}'))
    mixucbII_list = []
    mixucbII_rawreward_list = []
    q_mixUCBII_list = []
    for each_mixucbII_pkl in mixucbII_pkls:
        with open(os.path.join(result_root,f'mixucbII_results{mixucb_result_postfix}/{delta}', each_mixucbII_pkl), 'rb') as f:
            data = pickle.load(f)
            CR_mixucbII = data['CR_mixucbII']
            mixucbII_list.append(CR_mixucbII)
            # extract raw rewards from cumulative reward list.
            raw_rewards = [CR_mixucbII[i] - CR_mixucbII[i-1] if i > 0 else CR_mixucbII[i] for i in range(len(CR_mixucbII))]
            mixucbII_rawreward_list.append(raw_rewards)
            q_mixUCBII = data['q_mixucbII']
            q_mixUCBII_list.append(q_mixUCBII)

    mixucbIII_pkls = os.listdir(os.path.join(result_root,f'mixucbIII_results{mixucb_result_postfix}/{delta}'))
    mixucbIII_list = []
    mixucbIII_rawreward_list = []
    q_mixUCBIII_list = []
    for each_mixucbIII_pkl in mixucbIII_pkls:
        with open(os.path.join(result_root,f'mixucbIII_results{mixucb_result_postfix}/{delta}', each_mixucbIII_pkl), 'rb') as f:
            data = pickle.load(f)
            CR_mixucbIII = data['CR_mixucbIII']
            mixucbIII_list.append(CR_mixucbIII)
            # extract raw rewards from cumulative reward list.
            raw_rewards = [CR_mixucbIII[i] - CR_mixucbIII[i-1] if i > 0 else CR_mixucbIII[i] for i in range(len(CR_mixucbIII))]
            mixucbIII_rawreward_list.append(raw_rewards)
            q_mixUCBIII = data['q_mixucbIII']
            q_mixUCBIII_list.append(q_mixUCBIII)

    noisy_expert_pkls = os.listdir(os.path.join(result_root,f'noisy_expert_results'))
    noisy_expert_list = []
    for each_noisy_expert_pkl in noisy_expert_pkls:
        with open(os.path.join(result_root,'noisy_expert_results', each_noisy_expert_pkl), 'rb') as f:
            data = pickle.load(f)
            CR_NoisyExpert = data['CR_NoisyExpert']
            noisy_expert_list.append(CR_NoisyExpert)

    # Algorithm regret. Need to use individual rewards, then zero out query times, then recompute cumulative rewards.
    ARI_list = [np.array(perfect_expert_rawreward_list[0])*~np.array(q,dtype=bool) - np.array(l)*~np.array(q,dtype=bool) for (q,l) in zip(q_mixUCBI_list,mixucbI_rawreward_list)]
    ARI_list = [np.cumsum(l) for l in ARI_list]
    ARII_list = [np.array(perfect_expert_rawreward_list[0])*~np.array(q,dtype=bool) - np.array(l)*~np.array(q,dtype=bool) for (q,l) in zip(q_mixUCBII_list,mixucbII_rawreward_list)]
    ARII_list = [np.cumsum(l) for l in ARII_list]
    ARIII_list = [np.array(perfect_expert_rawreward_list[0])*~np.array(q,dtype=bool) - np.array(l)*~np.array(q,dtype=bool) for (q,l) in zip(q_mixUCBIII_list,mixucbIII_rawreward_list)]
    ARIII_list = [np.cumsum(l) for l in ARIII_list]

    # Reward regret.
    RRI_list = [np.array(perfect_expert_list[0]) - np.array(l) for l in mixucbI_list]
    RRII_list = [np.array(perfect_expert_list[0]) - np.array(l) for l in mixucbII_list]
    RRIII_list = [np.array(perfect_expert_list[0]) - np.array(l) for l in mixucbIII_list]

    CR_linucb_mean.append(np.mean(linucb_list, axis=0))
    CR_linucb_std.append(np.std(linucb_list, axis=0))
    CR_mixucbI_mean.append(np.mean(mixucbI_list, axis=0))
    CR_mixucbI_std.append(np.std(mixucbI_list, axis=0))
    CR_mixucbII_mean.append(np.mean(mixucbII_list, axis=0))
    CR_mixucbII_std.append(np.std(mixucbII_list, axis=0))
    CR_mixucbIII_mean.append(np.mean(mixucbIII_list, axis=0))
    CR_mixucbIII_std.append(np.std(mixucbIII_list, axis=0))
    CR_NoisyExpert_mean.append(np.mean(noisy_expert_list, axis=0))
    CR_NoisyExpert_std.append(np.std(noisy_expert_list, axis=0))
    CR_PerfectExpert_mean.append(np.mean(perfect_expert_list, axis=0))
    CR_PerfectExpert_std.append(np.std(perfect_expert_list, axis=0))

    q_mixUCBI_mean.append(np.mean(q_mixUCBI_list, axis=0))
    q_mixUCBI_std.append(np.std(q_mixUCBI_list, axis=0))
    q_mixUCBII_mean.append(np.mean(q_mixUCBII_list, axis=0))
    q_mixUCBII_std.append(np.std(q_mixUCBII_list, axis=0))
    q_mixUCBIII_mean.append(np.mean(q_mixUCBIII_list, axis=0))
    q_mixUCBIII_std.append(np.std(q_mixUCBIII_list, axis=0))

    # Cumulative queries.
    CQ_mixucbI_mean.append(np.mean([np.cumsum(q) for q in q_mixUCBI_list],axis=0))
    CQ_mixucbI_std.append(np.std([np.cumsum(q) for q in q_mixUCBI_list],axis=0))
    CQ_mixucbII_mean.append(np.mean([np.cumsum(q) for q in q_mixUCBII_list],axis=0))
    CQ_mixucbII_std.append(np.std([np.cumsum(q) for q in q_mixUCBII_list],axis=0))
    CQ_mixucbIII_mean.append(np.mean([np.cumsum(q) for q in q_mixUCBIII_list],axis=0))
    CQ_mixucbIII_std.append(np.std([np.cumsum(q) for q in q_mixUCBIII_list],axis=0))

    # AR and RR.
    AR_mixucbI_mean.append(np.mean(ARI_list, axis=0))
    AR_mixucbI_std.append(np.std(ARI_list, axis=0))
    AR_mixucbII_mean.append(np.mean(ARII_list, axis=0))
    AR_mixucbII_std.append(np.std(ARII_list, axis=0))
    AR_mixucbIII_mean.append(np.mean(ARIII_list, axis=0))
    AR_mixucbIII_std.append(np.std(ARIII_list, axis=0))

    RR_mixucbI_mean.append(np.mean(RRI_list, axis=0))
    RR_mixucbI_std.append(np.std(RRI_list, axis=0))
    RR_mixucbII_mean.append(np.mean(RRII_list, axis=0))
    RR_mixucbII_std.append(np.std(RRII_list, axis=0))
    RR_mixucbIII_mean.append(np.mean(RRIII_list, axis=0))
    RR_mixucbIII_std.append(np.std(RRIII_list, axis=0))

    cumulative_rewards = {
        'LinUCB': CR_linucb_mean,
        f'MixUCB-I ($\\Delta = {delta}$)': CR_mixucbI_mean,
        f'MixUCB-II ($\\Delta = {delta}$)': CR_mixucbII_mean,
        f'MixUCB-III ($\\Delta = {delta}$)': CR_mixucbIII_mean,
        'NoisyExpert': CR_NoisyExpert_mean,
        'PerfectExpert': CR_PerfectExpert_mean,
    }
    cumulative_rewards_std = {
        'LinUCB': CR_linucb_std,
        f'MixUCB-I ($\\Delta = {delta}$)': CR_mixucbI_std,
        f'MixUCB-II ($\\Delta = {delta}$)': CR_mixucbII_std,
        f'MixUCB-III ($\\Delta = {delta}$)': CR_mixucbIII_std,
        'NoisyExpert': CR_NoisyExpert_std,
        'PerfectExpert': CR_PerfectExpert_std,
    }

    q_mean = {
        'MixUCB-I': q_mixUCBI_mean,
        'MixUCB-II': q_mixUCBII_mean,
        'MixUCB-III': q_mixUCBIII_mean,
    }

    q_std = {
        'MixUCB-I': q_mixUCBI_std,
        'MixUCB-II': q_mixUCBII_std,
        'MixUCB-III': q_mixUCBIII_std,
    }

    cq_mean = {
        'MixUCB-I': CQ_mixucbI_mean,
        'MixUCB-II': CQ_mixucbII_mean,
        'MixUCB-III': CQ_mixucbIII_mean,
    }

    cq_std = {
        'MixUCB-I': CQ_mixucbI_std,
        'MixUCB-II': CQ_mixucbII_std,
        'MixUCB-III': CQ_mixucbIII_std,
    }

    ar = {
        f'MixUCB-I ($\\Delta = {delta}$)': AR_mixucbI_mean,
        f'MixUCB-II ($\\Delta = {delta}$)': AR_mixucbII_mean,
        f'MixUCB-III ($\\Delta = {delta}$)': AR_mixucbIII_mean,
    }

    ar_std = {
        f'MixUCB-I ($\\Delta = {delta}$)': AR_mixucbI_std,
        f'MixUCB-II ($\\Delta = {delta}$)': AR_mixucbII_std,
        f'MixUCB-III ($\\Delta = {delta}$)': AR_mixucbIII_std,
    }

    rr = {
        f'MixUCB-I ($\\Delta = {delta}$)': RR_mixucbI_mean,
        f'MixUCB-II ($\\Delta = {delta}$)': RR_mixucbII_mean,
        f'MixUCB-III ($\\Delta = {delta}$)': RR_mixucbIII_mean,
    }

    rr_std = {
        f'MixUCB-I ($\\Delta = {delta}$)': RR_mixucbI_std,
        f'MixUCB-II ($\\Delta = {delta}$)': RR_mixucbII_std,
        f'MixUCB-III ($\\Delta = {delta}$)': RR_mixucbIII_std,
    }

    # Add queries to this plot.
    fig, axs = plt.subplots(2, 1, figsize=(8, 16))
    plot_average_rewards([axs[0]], cumulative_rewards, cumulative_rewards_std)
    plot_cumulative_queries([axs[1]], cq_mean, cq_std, [delta])
    plt.tight_layout()
    fig.savefig(os.path.join(Figure_dir, f'six_baselines_avgr.png'), format='jpg', dpi=300, bbox_inches='tight')
    
    fig, axs = plt.subplots(1, 1, figsize=(8, 8))
    plot_cumulative_rewards([axs], cumulative_rewards, cumulative_rewards_std)
    fig.savefig(os.path.join(Figure_dir, f'six_baselines_cr.png'), format='jpg', dpi=300, bbox_inches='tight')
    plt.tight_layout()

    # Add a algorithm regret plot, which is difference between perfect expert and querying algorithm,
    # but is 0 when we query. should be cumulative as well.
    # For now we can compute this for just MixUCB-I, MixUCB-II, MixUCB-III.
    fig, axs = plt.subplots(2, 1, figsize=(8, 16))
    plot_cumulative_rewards([axs[0]], ar, ar_std,ylabel="Algorithm Regret")
    plot_cumulative_queries([axs[1]], cq_mean, cq_std, [delta])
    plt.tight_layout()
    fig.savefig(os.path.join(Figure_dir, f'six_baselines_ar.png'), format='jpg', dpi=300, bbox_inches='tight')

    # Add a reward regret plot, which is difference between perfect expert and querying algorithm.
    fig, axs = plt.subplots(2, 1, figsize=(8, 16))
    plot_cumulative_rewards([axs[0]], rr, rr_std,ylabel="Reward Regret")
    plot_cumulative_queries([axs[1]], cq_mean, cq_std, [delta])
    plt.tight_layout()
    fig.savefig(os.path.join(Figure_dir, f'six_baselines_rr.png'), format='jpg', dpi=300, bbox_inches='tight')

    plt.close()

def plot_three_mixucbs(Figure_dir='Figures', result_postfix="", result_root=''):
    os.makedirs(Figure_dir, exist_ok=True)
    CR_mixucbI_mean = []
    CR_mixucbI_std = []
    CR_mixucbII_mean = []
    CR_mixucbII_std = []
    CR_mixucbIII_mean = []
    CR_mixucbIII_std = []

    q_mixUCBI_mean = []
    q_mixUCBI_std = []
    q_mixUCBII_mean = []
    q_mixUCBII_std = []
    q_mixUCBIII_mean = []
    q_mixUCBIII_std = []

    # cumulative queries (CQ)
    CQ_mixucbI_mean = []
    CQ_mixucbI_std = []
    CQ_mixucbII_mean = []
    CQ_mixucbII_std = []
    CQ_mixucbIII_mean = []
    CQ_mixucbIII_std = []

    TotalQ_mixucbI_mean = []
    TotalQ_mixucbI_std = []
    TotalQ_mixucbII_mean = []
    TotalQ_mixucbII_std = []
    TotalQ_mixucbIII_mean = []
    TotalQ_mixucbIII_std = []

    # delta_values = [0.2, 0.5, 1.,2., 5.]
    delta_values = [0.2, 0.5, 1.]
    for each_delta in delta_values:
        mixucbI_pkls = os.listdir(os.path.join(result_root,f'mixucbI_results{result_postfix}','{}'.format(each_delta)))
        mixucbI_list = []
        mixUCBI_q_list = []
        mixucbI_list_totalQ = []
        for each_mixucbI_pkl in mixucbI_pkls:
            with open(os.path.join(result_root,f'mixucbI_results{result_postfix}','{}'.format(each_delta), each_mixucbI_pkl), 'rb') as f:
                data = pickle.load(f)
                CR_mixucbI = data['CR_mixucbI']
                mixucbI_list.append(CR_mixucbI)
                TotalQ_mixUCBI = data['TotalQ_mixucbI']
                mixucbI_list_totalQ.append(TotalQ_mixUCBI)
                q_mixUCBI = data['q_mixucbI']
                mixUCBI_q_list.append(q_mixUCBI)

        mixucbII_pkls = os.listdir(os.path.join(result_root,f'mixucbII_results{result_postfix}','{}'.format(each_delta)))
        mixucbII_list = []
        mixUCBII_q_list = []
        mixucbII_list_totalQ = []
        for each_mixucbII_pkl in mixucbII_pkls:
            with open(os.path.join(result_root,f'mixucbII_results{result_postfix}','{}'.format(each_delta), each_mixucbII_pkl), 'rb') as f:
                data = pickle.load(f)
                CR_mixucbII = data['CR_mixucbII']
                mixucbII_list.append(CR_mixucbII)
                TotalQ_mixUCBII = data['TotalQ_mixucbII']
                mixucbII_list_totalQ.append(TotalQ_mixUCBII)
                q_mixUCBII = data['q_mixucbII']
                mixUCBII_q_list.append(q_mixUCBII)

        mixucbIII_pkls = os.listdir(os.path.join(result_root,f'mixucbIII_results{result_postfix}','{}'.format(each_delta)))
        mixucbIII_list = []
        mixUCBIII_q_list = []
        mixucbIII_list_totalQ = []
        for each_mixucbIII_pkl in mixucbIII_pkls:
            with open(os.path.join(result_root,f'mixucbIII_results{result_postfix}','{}'.format(each_delta), each_mixucbIII_pkl), 'rb') as f:
                data = pickle.load(f)
                CR_mixucbIII = data['CR_mixucbIII']
                mixucbIII_list.append(CR_mixucbIII)
                TotalQ_mixUCBIII = data['TotalQ_mixucbIII']
                mixucbIII_list_totalQ.append(TotalQ_mixUCBIII)
                q_mixUCBIII = data['q_mixucbIII']
                mixUCBIII_q_list.append(q_mixUCBIII)


        CR_mixucbI_mean.append(np.mean(mixucbI_list, axis=0))
        CR_mixucbI_std.append(np.std(mixucbI_list, axis=0))
        CR_mixucbII_mean.append(np.mean(mixucbII_list, axis=0))
        CR_mixucbII_std.append(np.std(mixucbII_list, axis=0))
        CR_mixucbIII_mean.append(np.mean(mixucbIII_list, axis=0))
        CR_mixucbIII_std.append(np.std(mixucbIII_list, axis=0))

        q_mixUCBI_mean.append(np.mean(mixUCBI_q_list, axis=0))
        q_mixUCBI_std.append(np.std(mixUCBI_q_list, axis=0))
        q_mixUCBII_mean.append(np.mean(mixUCBII_q_list, axis=0))
        q_mixUCBII_std.append(np.std(mixUCBII_q_list, axis=0))
        q_mixUCBIII_mean.append(np.mean(mixUCBIII_q_list, axis=0))
        q_mixUCBIII_std.append(np.std(mixUCBIII_q_list, axis=0))

        CQ_mixucbI_mean.append(np.mean([np.cumsum(q) for q in mixUCBI_q_list],axis=0))
        CQ_mixucbI_std.append(np.std([np.cumsum(q) for q in mixUCBI_q_list],axis=0))
        CQ_mixucbII_mean.append(np.mean([np.cumsum(q) for q in mixUCBII_q_list],axis=0))
        CQ_mixucbII_std.append(np.std([np.cumsum(q) for q in mixUCBII_q_list],axis=0))
        CQ_mixucbIII_mean.append(np.mean([np.cumsum(q) for q in mixUCBIII_q_list],axis=0))
        CQ_mixucbIII_std.append(np.std([np.cumsum(q) for q in mixUCBIII_q_list],axis=0))

        TotalQ_mixucbI_mean.append(np.mean(mixucbI_list_totalQ))
        TotalQ_mixucbI_std.append(np.std(mixucbI_list_totalQ))
        TotalQ_mixucbII_mean.append(np.mean(mixucbII_list_totalQ))
        TotalQ_mixucbII_std.append(np.std(mixucbII_list_totalQ))
        TotalQ_mixucbIII_mean.append(np.mean(mixucbIII_list_totalQ))
        TotalQ_mixucbIII_std.append(np.std(mixucbIII_list_totalQ))

    cumulative_rewards = {
        'MixUCB-I': CR_mixucbI_mean,
        'MixUCB-II': CR_mixucbII_mean,
        'MixUCB-III': CR_mixucbIII_mean,
    }
    cumulative_rewards_std = {
        'MixUCB-I': CR_mixucbI_std,
        'MixUCB-II': CR_mixucbII_std,
        'MixUCB-III': CR_mixucbIII_std,
    }

    q_mean = {
        'MixUCB-I': q_mixUCBI_mean,
        'MixUCB-II': q_mixUCBII_mean,
        'MixUCB-III': q_mixUCBIII_mean,
    }

    q_std = {
        'MixUCB-I': q_mixUCBI_std,
        'MixUCB-II': q_mixUCBII_std,
        'MixUCB-III': q_mixUCBIII_std,
    }

    cq_mean = {
        'MixUCB-I': CQ_mixucbI_mean,
        'MixUCB-II': CQ_mixucbII_mean,
        'MixUCB-III': CQ_mixucbIII_mean,
    }

    cq_std = {
        'MixUCB-I': CQ_mixucbI_std,
        'MixUCB-II': CQ_mixucbII_std,
        'MixUCB-III': CQ_mixucbIII_std,
    }

    print(f"Deltas: {delta_values}")
    print(f'TotalQ_mixucbI_mean: {np.array(TotalQ_mixucbI_mean)}')
    print(f'TotalQ_mixucbII_mean: {np.array(TotalQ_mixucbII_mean)}')
    print(f'TotalQ_mixucbIII_mean: {np.array(TotalQ_mixucbIII_mean)}')
    
    # Create a 1 by m grid of subplots for average rewards
    fig, axs = plt.subplots(1, len(delta_values), figsize=(18, 3))
    plot_average_rewards(axs, cumulative_rewards, cumulative_rewards_std, delta_values)
    plt.tight_layout()
    fig.savefig(os.path.join(Figure_dir, f'three_mixucbs_avgr.png'), format='jpg', dpi=300, bbox_inches='tight')

    fig, axs = plt.subplots(1, len(delta_values), figsize=(18, 3))
    plot_cumulative_rewards(axs, cumulative_rewards, cumulative_rewards_std, delta_values)
    plt.tight_layout()
    fig.savefig(os.path.join(Figure_dir, f'three_mixucbs_cr.png'), format='jpg', dpi=300, bbox_inches='tight')

    fig,axs = plt.subplots(1,len(delta_values),figsize=(18,3))
    plot_cumulative_queries(axs, cq_mean, cq_std, delta_values)
    plt.tight_layout()
    fig.savefig(os.path.join(Figure_dir, f'three_mixucbs_q.png'), format='jpg', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    # 2024-10-6 experiment 2
    # mixucb_postfix="_3" # corresponds to a setting, see tune_mixUCB.py
    # 2024-10-6 experiment 3
    # mixucb_postfix="_7"
    # mixucb_postfix="_0"  # for g2 reproducibility experiment, 10/6/24
    # 2024-10-07 for re-plotting w/ number of queries
    # mixucb_postfix="_7"
    # result_root = 'gridsearchpart3_20241006'
    # 2024-10-07 g2 experiment
    # mixucb_postfix="_19"
    # result_root='g2highertemp_20241007'
    # Temp 1.0 and Temp 5.0 experiments, 10/7
    # Temp 1.0
    # mixucb_postfix="_temp1.0_2"
    # result_root="g2temp1.0_20241007"
    # temp 5.0
    # mixucb_postfix="_temp5.0_3"
    # result_root="g2temp5.0_20241007"
    # Higher beta experiments, 10/8
    # Temp 1.0
    # mixucb_postfix="_temp1.0_5"
    # result_root="g2temp1.0_higherbeta_20241008"
    # Temp 5.0
    # mixucb_postfix="_temp5.0_6"
    # result_root="g2temp5.0_higherbeta_20241008"
    # Linear reward oracle experiments, 10/8
    # Temp 1.0
    # mixucb_postfix="_temp1.0_7"
    # result_root="g2temp1.0_linearreward_20241008"
    # Figure_dir = f'Figures/{result_root}'
    # Temp 5.0

    # Experiments 10/9 (expected reward oracle)
    # Temp 1.0
    # mixucb_postfix="_temp1.0_10"
    # result_root="g2temp1.0_explinearoracle_20241009"
    # Temp 5.0
    # mixucb_postfix="_temp5.0_10"
    # result_root="g2temp5.0_explinearoracle_20241009"

    # 10/9, synthetic.
    # Temp 5.0
    # mixucb_postfix="_temp5.0_1"
    # result_root="synthetictemp5.0_20241009"

    # Temp 5.0
    # mixucb_postfix="_temp5.0_7"
    # result_root="synthetictemp5.0_20241009_5"

    # Temp 5.0, Alpha 1
    # (need to use lambda=0.001 setting)
    # mixucb_postfix="_temp5.0_alpha1.0_0"
    # result_root="syntheticalpha1_20241010"

    # Temp 1.0, Alpha 0.01
    # mixucb_postfix="_temp5.0_alpha0.01_0"
    # result_root="syntheticalpha0.01_20241010"

    # Temp 5.0, Alpha 0.75
    # mixucb_postfix="_temp5.0_alpha0.75_0"
    # result_root="syntheticalpha0.75_20241010_2"

    # Temp 5.0, Alpha 0.75 again.
    # mixucb_postfix="_temp5.0_alpha0.75_0"   #beta=50
    # # mixucb_postfix="_temp5.0_alpha0.75_2" #beta=200
    # result_root="syntheticalpha0.75_20241010_3"
    
    # Again.
    # NOTE: must use a setting with lambda=0.001
    # mixucb_postfix="_temp5.0_alpha0.75_0"   #beta=10
    # # mixucb_postfix="_temp5.0_alpha0.75_1"   #beta=10
    # result_root="syntheticalpha0.75_20241010_4"

    # Back to SPANet
    mixucb_postfix = "_temp5.0_alpha0.1_0"  # beta=5000
    result_root = "spanettemp5.0_20241010"

    Figure_dir = f'Figures/{result_root}'

    plot_three_mixucbs(Figure_dir=Figure_dir, result_postfix=mixucb_postfix,result_root=result_root)
    # NOTE: using a fixed value of delta.
    # delta=0.5
    # for delta in [0.2, 0.5, 1., 2., 5.]:
    # for delta in [0.2, 0.5, 1.]:
    # for delta in [0.2, 0.5, 0.75, 1.]:
    # for delta in [0.2, 0.5]:
    for delta in [0.2, 0.5, 1.]:
        plot_six_baselines(Figure_dir=f"{Figure_dir}_delta{delta}", mixucb_result_postfix=mixucb_postfix,delta=delta,result_root=result_root)

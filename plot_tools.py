import matplotlib
# matplotlib.use('TKAgg') # for g2, comment me out.
import matplotlib.pyplot as plt
import os
import pickle
import numpy as np
import argparse
from icecream import ic
import seaborn as sns

def extract_from(filename):
    print(filename)
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        reward_per_time = data['reward_per_time']
        try:
            query_per_time = data['query_per_time']
        except KeyError:
            query_per_time = None
        action_per_time = data['action_per_time']
    return reward_per_time, query_per_time, action_per_time

def plot_mixucbs(Figure_dir='Figures', result_postfix="", result_root='', data_name='', seeds=[42]):
    os.makedirs(Figure_dir, exist_ok=True)
    modes = {'lin':['LinUCB'], 
             'mixI':['MixUCB-I'], 
             'mixII':['MixUCB-II'],
             'mixIII':['MixUCB-III'], 
             'sq_oracle':['Lin Oracle (Regression)'], 
             'lr_oracle':['Lin Oracle (Classification)'],
             'perfect_exp':['Perfect Expert'],
             'noisy_exp':['Noisy Expert']}
    # If dataset is either 'heart_disease' or 'MedNIST', we won't show perfect expert or noisy expert.
    if data_name in ['heart_disease', 'MedNIST']:
        modes.pop('perfect_exp', None)
        modes.pop('noisy_exp', None)
    colors_list = sns.color_palette('colorblind', n_colors=len(modes.keys()))
    for mode, color in zip(modes.keys(), colors_list):
        modes[mode].append(color)
    marker_list = [".", "o", "v", "s", "+", "D", '1', 'p']    # shapes for each mode.
    assert len(marker_list) >= len(modes.keys())
    for mode, marker in zip(modes.keys(), marker_list):
        modes[mode].append(marker)
    markerevery = 30
    rewards = {mode: {} for mode in modes.keys()}       # keying is mode/seed.
    queries = {mode: {} for mode in modes.keys()}
    actions = {mode: {} for mode in modes.keys()}
    # data containers for each of the 3 subplots/metrics.
    # (i.e. for each of the modes/algorithms, we have 3 signals of interest.)
    cumulative_rewards = {mode: {} for mode in modes.keys()}       # keying is mode/seed.
    cumulative_queries = {mode: {} for mode in modes.keys()}
    avg_reward_noquery = {mode: {} for mode in modes.keys()}

    delta_values = None

    ## Data gathering: loop across seeds.
    ## Data containers (rewards, queries, actions) should contain seed as a key.

    for seed in seeds:
        # modes are algorithms.
        for mode in modes.keys():
            if mode in ['sq_oracle', 'lr_oracle', 'perfect_exp', 'noisy_exp']:
                rewards[mode][seed] = []
                actions[mode][seed] = []
                dirname = os.path.join(result_root, f'seed_{seed:02d}', f'{mode}_results{result_postfix}')
                pkls = os.listdir(dirname)
                for pklname in pkls:
                    reward_per_time, _, action_per_time = extract_from(os.path.join(dirname,pklname))
                    rewards[mode][seed].append(reward_per_time)
                    actions[mode][seed].append(action_per_time)
            elif mode == 'lin':
                rewards[mode][seed] = []
                queries[mode][seed] = []
                actions[mode][seed] = []
                dirname = os.path.join(result_root, f'seed_{seed:02d}', f'{mode}_ucb_results{result_postfix}', '{}'.format(0))
                pkls = os.listdir(dirname)
                for pklname in pkls:
                    reward_per_time, query_per_time, action_per_time = extract_from(os.path.join(dirname,pklname))
                    rewards[mode][seed].append(reward_per_time)
                    queries[mode][seed].append(query_per_time)
                    actions[mode][seed].append(action_per_time)
            else:
                rewards[mode][seed] = {}
                queries[mode][seed] = {}
                actions[mode][seed] = {}
                new_delta_values = np.sort(os.listdir(os.path.join(result_root, f'seed_{seed:02d}', f'{mode}_ucb_results{result_postfix}')))
                if delta_values is not None:
                    assert (delta_values == new_delta_values).all()
                else:
                    delta_values = new_delta_values
                for each_delta in delta_values:
                    print(each_delta)
                    rewards[mode][seed][each_delta] = []
                    queries[mode][seed][each_delta] = []
                    actions[mode][seed][each_delta] = []
                    dirname = os.path.join(result_root, f'seed_{seed:02d}', f'{mode}_ucb_results{result_postfix}', '{}'.format(each_delta))
                    pkls = os.listdir(dirname)
                    for pklname in pkls:
                        reward_per_time, query_per_time, action_per_time = extract_from(os.path.join(dirname,pklname))
                        rewards[mode][seed][each_delta].append(reward_per_time)
                        queries[mode][seed][each_delta].append(query_per_time)
                        actions[mode][seed][each_delta].append(action_per_time)

        def accum(list_of_lists):
            try:
                return np.array([np.cumsum(l) for l in list_of_lists])
            except ValueError as e:
                print(f"Error in accumulating lists: {e}")
                print([np.cumsum(l) for l in list_of_lists])
                raise ValueError
        def masked_avg(list_of_values, list_of_masks):
            ret = []
            for vals, mask in zip(list_of_values, list_of_masks):
                mask = (1-np.array(mask)).astype(int)

                temp = np.zeros_like(vals)
                for i in range(len(temp)):
                    if mask[i]:
                        temp[i] = vals[i]
                ret.append(np.cumsum(temp) / np.cumsum(mask))

            return np.array(ret)
        # cumulative reward over time
        for mode in modes.keys():
            if mode in ['sq_oracle', 'lr_oracle', 'perfect_exp', 'noisy_exp']:
                cumulative_rewards[mode][seed] = np.mean(accum(rewards[mode][seed]), axis=0)
                avg_reward_noquery[mode][seed] = np.mean(masked_avg(rewards[mode][seed],np.zeros_like(rewards[mode][seed])), axis=0)
            elif mode == 'lin':
                cumulative_rewards[mode][seed] = np.mean(accum(rewards[mode][seed]), axis=0)
                cumulative_queries[mode][seed] = np.mean(accum(queries[mode][seed]), axis=0)
                avg_reward_noquery[mode][seed] = np.mean(masked_avg(rewards[mode][seed], queries[mode][seed]), axis=0)
            else:
                cumulative_rewards[mode][seed] = {}
                cumulative_queries[mode][seed] = {}
                avg_reward_noquery[mode][seed] = {}
                for each_delta in delta_values:
                    cumulative_rewards[mode][seed][each_delta] = np.mean(accum(rewards[mode][seed][each_delta]), axis=0)
                    cumulative_queries[mode][seed][each_delta] = np.mean(accum(queries[mode][seed][each_delta]), axis=0)
                    avg_reward_noquery[mode][seed][each_delta] = np.mean(masked_avg(rewards[mode][seed][each_delta], queries[mode][seed][each_delta]), axis=0)

    print(f"Seeds: {seeds}")

    # plot the data.
    ## [meaning, for each mode/algorithm pair, plot the median and the quartiles (or mean and std) over the seeds.]
    fig_list, axs_list = [], []
    for i, each_delta in enumerate(delta_values):
        num_metrics_to_plot = 3 # Cumulative Reward, Cumulative Queries, Average Autonomous Reward
        # Create a figure with 3 subplots (1 row, 3 columns).
        fig, axs = plt.subplots(1, num_metrics_to_plot, figsize=(10, 3))
        fig_list.append(fig)
        axs_list.append(axs)
    
    # Plot 1: Cumulative Reward

    non_mixucb_labels = ['sq_oracle', 'lr_oracle', 'perfect_exp', 'noisy_exp', 'lin']
    mixucb_labels = ['mixI', 'mixII', 'mixIII']

    for i, each_delta in enumerate(delta_values):
        ax = axs_list[i][0]
        for mode in modes.keys():
            if mode in non_mixucb_labels:
                # Calculate mean + std of cumulative_rewards over the seeds.
                mean_cumulative_rewards = np.mean([cumulative_rewards[mode][seed] for seed in seeds], axis=0)
                std_cumulative_rewards = np.std([cumulative_rewards[mode][seed] for seed in seeds], axis=0)
                ax.plot(mean_cumulative_rewards, label=modes[mode][0], color=modes[mode][1], marker=modes[mode][2], markevery=markerevery)
                ax.fill_between(range(len(mean_cumulative_rewards)), mean_cumulative_rewards - std_cumulative_rewards, mean_cumulative_rewards + std_cumulative_rewards, \
                                color=modes[mode][1], alpha=0.2)
            elif mode in mixucb_labels:
                # Calculate mean + std of cumulative_rewards over the seeds for each delta.
                mean_cumulative_rewards = np.mean([cumulative_rewards[mode][seed][each_delta] for seed in seeds], axis=0)
                std_cumulative_rewards = np.std([cumulative_rewards[mode][seed][each_delta] for seed in seeds], axis=0)
                ax.plot(mean_cumulative_rewards, label=modes[mode][0], color=modes[mode][1], marker=modes[mode][2], markevery=markerevery)
                ax.fill_between(range(len(mean_cumulative_rewards)), mean_cumulative_rewards - std_cumulative_rewards, mean_cumulative_rewards + std_cumulative_rewards, \
                                color=modes[mode][1], alpha=0.2)
        ax.set_title('Cumulative Reward ($\Delta={}$)'.format(each_delta))
        ax.set_xlabel('time steps')

    # Plot 2: Cumulative Queries
    for i, each_delta in enumerate(delta_values):
        ax = axs_list[i][2]
        for mode in mixucb_labels:
            # Calculate mean + std of cumulative_queries over the seeds.
            mean_cumulative_queries = np.mean([cumulative_queries[mode][seed][each_delta] for seed in seeds], axis=0)
            std_cumulative_queries = np.std([cumulative_queries[mode][seed][each_delta] for seed in seeds], axis=0)
            ax.plot(mean_cumulative_queries, label=modes[mode][0], color=modes[mode][1], marker=modes[mode][2], markevery=markerevery)
            ax.fill_between(range(len(mean_cumulative_queries)), mean_cumulative_queries - std_cumulative_queries, mean_cumulative_queries + std_cumulative_queries, \
                            color=modes[mode][1], alpha=0.2)
        ax.set_title('Cumulative Queries ($\Delta={}$)'.format(each_delta))
        ax.set_xlabel('time steps')

    # Plot 3: Average Autonomous Reward
    for i, each_delta in enumerate(delta_values):
        ax = axs_list[i][1]
        for mode in modes.keys():
            if mode in non_mixucb_labels:
                # Calculate mean + std of avg_reward_noquery over the seeds.
                mean_avg_reward_noquery = np.mean([avg_reward_noquery[mode][seed] for seed in seeds], axis=0)
                std_avg_reward_noquery = np.std([avg_reward_noquery[mode][seed] for seed in seeds], axis=0)
                ax.plot(mean_avg_reward_noquery, label=modes[mode][0], color=modes[mode][1], marker=modes[mode][2], markevery=markerevery)
                ax.fill_between(range(len(mean_avg_reward_noquery)), mean_avg_reward_noquery - std_avg_reward_noquery, mean_avg_reward_noquery + std_avg_reward_noquery, \
                                color=modes[mode][1], alpha=0.2)
            elif mode in mixucb_labels:
                # Calculate mean + std of avg_reward_noquery over the seeds for each delta.
                mean_avg_reward_noquery = np.mean([avg_reward_noquery[mode][seed][each_delta] for seed in seeds], axis=0)
                std_avg_reward_noquery = np.std([avg_reward_noquery[mode][seed][each_delta] for seed in seeds], axis=0)
                ax.plot(mean_avg_reward_noquery, label=modes[mode][0], color=modes[mode][1], marker=modes[mode][2], markevery=markerevery)
                ax.fill_between(range(len(mean_avg_reward_noquery)), mean_avg_reward_noquery - std_avg_reward_noquery, mean_avg_reward_noquery + std_avg_reward_noquery, \
                                color=modes[mode][1], alpha=0.2)
        ax.set_title('Average Autonomous Reward ($\Delta={}$)'.format(each_delta))
        ax.set_xlabel('time steps')

    handles, labels = axs_list[0][0].get_legend_handles_labels()
    for i, each_delta in enumerate(delta_values):
        fig = fig_list[i]
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.0),
                  fancybox=True, shadow=True, ncol=len(labels), fontsize='small')
        fig.tight_layout()
        fig.savefig(os.path.join(Figure_dir, f'delta{each_delta}.pdf'), format='pdf', dpi=300, bbox_inches='tight')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for processing data.")
    parser.add_argument("--data_name", type=str, default='', help="Name of the dataset to use.")
    parser.add_argument("--seeds", nargs='+', type=int, default=[42], help="List of seeds to use.")
    args = parser.parse_args()
    data_name = args.data_name

    result_root = data_name
    Figure_dir = f'Figures/{result_root}'
    seeds= args.seeds
    os.makedirs(Figure_dir, exist_ok=True)
    plot_mixucbs(Figure_dir=Figure_dir, result_root=result_root, data_name=data_name, seeds=seeds)

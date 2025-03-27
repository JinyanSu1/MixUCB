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

def plot_mixucbs(Figure_dir='Figures', result_postfix="", result_root='', data_name=''):
    os.makedirs(Figure_dir, exist_ok=True)
    colors_list = sns.color_palette('colorblind', n_colors=6)
    modes = {'lin':['LinUCB'], 
             'mixI':['MixUCB-I'], 
             'mixII':['MixUCB-II'],
             'mixIII':['MixUCB-III'], 
             'sq_oracle':['Lin Oracle (Regression)'], 
             'lr_oracle':['Lin Oracle (Classification)']}
    for mode, color in zip(modes.keys(), colors_list):
        modes[mode].append(color)
    marker_list = [".", "o", "v", "s", "+", "D"]
    assert len(marker_list) >= len(modes.keys())
    for mode, marker in zip(modes.keys(), marker_list):
        modes[mode].append(marker)
    markerevery = 30
    rewards = {}
    queries = {}
    actions = {}

    delta_values = None

    for mode in modes.keys():
        if mode in ['sq_oracle', 'lr_oracle']:
            rewards[mode] = []
            actions[mode] = []
            dirname = os.path.join(result_root, f'{mode}_results{result_postfix}')
            pkls = os.listdir(dirname)
            for pklname in pkls:
                reward_per_time, _, action_per_time = extract_from(os.path.join(dirname,pklname))
                rewards[mode].append(reward_per_time)
                actions[mode].append(action_per_time)
        elif mode == 'lin':
            rewards[mode] = []
            queries[mode] = []
            actions[mode] = []
            dirname = os.path.join(result_root, f'{mode}_ucb_results{result_postfix}', '{}'.format(0))
            pkls = os.listdir(dirname)
            for pklname in pkls:
                reward_per_time, query_per_time, action_per_time = extract_from(os.path.join(dirname,pklname))
                rewards[mode].append(reward_per_time)
                queries[mode].append(query_per_time)
                actions[mode].append(action_per_time)
        else:
            rewards[mode] = {}
            queries[mode] = {}
            actions[mode] = {}
            new_delta_values = np.sort(os.listdir(os.path.join(result_root, f'{mode}_ucb_results{result_postfix}')))
            if delta_values is not None:
                assert (delta_values == new_delta_values).all()
            else:
                delta_values = new_delta_values
            for each_delta in delta_values:
                print(each_delta)
                rewards[mode][each_delta] = []
                queries[mode][each_delta] = []
                actions[mode][each_delta] = []
                dirname = os.path.join(result_root, f'{mode}_ucb_results{result_postfix}', '{}'.format(each_delta))
                pkls = os.listdir(dirname)
                for pklname in pkls:
                    reward_per_time, query_per_time, action_per_time = extract_from(os.path.join(dirname,pklname))
                    rewards[mode][each_delta].append(reward_per_time)
                    queries[mode][each_delta].append(query_per_time)
                    actions[mode][each_delta].append(action_per_time)

    cumulative_rewards = {}
    cumulative_queries = {}
    avg_reward_noquery = {}
    def accum(list_of_lists):
        return np.array([np.cumsum(l) for l in list_of_lists])
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
        if mode in ['sq_oracle', 'lr_oracle']:
            cumulative_rewards[mode] = np.mean(accum(rewards[mode]), axis=0)
            avg_reward_noquery[mode] = np.mean(masked_avg(rewards[mode],np.zeros_like(rewards[mode])), axis=0)
        elif mode == 'lin':
            cumulative_rewards[mode] = np.mean(accum(rewards[mode]), axis=0)
            cumulative_queries[mode] = np.mean(accum(queries[mode]), axis=0)
            avg_reward_noquery[mode] = np.mean(masked_avg(rewards[mode], queries[mode]), axis=0)
        else:
            cumulative_rewards[mode] = {}
            cumulative_queries[mode] = {}
            avg_reward_noquery[mode] = {}
            for each_delta in delta_values:
                cumulative_rewards[mode][each_delta] = np.mean(accum(rewards[mode][each_delta]), axis=0)
                cumulative_queries[mode][each_delta] = np.mean(accum(queries[mode][each_delta]), axis=0)
                avg_reward_noquery[mode][each_delta] = np.mean(masked_avg(rewards[mode][each_delta], queries[mode][each_delta]), axis=0)

    fig_list, axs_list = [], []
    for i, each_delta in enumerate(delta_values):
        fig, axs = plt.subplots(1, len(delta_values), figsize=(10, 3))
        fig_list.append(fig)
        axs_list.append(axs)
    
    for i, each_delta in enumerate(delta_values):
        ax = axs_list[i][0]
        for mode in ['sq_oracle', 'lr_oracle', 'lin']:
            ax.plot(cumulative_rewards[mode], label=modes[mode][0], color=modes[mode][1], marker=modes[mode][2], markevery=markerevery)
        for mode in ['mixI','mixII', 'mixIII']:
            ax.plot(cumulative_rewards[mode][each_delta], label=modes[mode][0], color=modes[mode][1], marker=modes[mode][2], markevery=markerevery)
        ax.set_title('Cumulative Reward ($\Delta={}$)'.format(each_delta))
        ax.set_xlabel('time steps')

    for i, each_delta in enumerate(delta_values):
        ax = axs_list[i][2]
        for mode in ['mixI','mixII', 'mixIII']:
            ax.plot(cumulative_queries[mode][each_delta], label=modes[mode][0], color=modes[mode][1], marker=modes[mode][2], markevery=markerevery)
        ax.set_title('Cumulative Queries ($\Delta={}$)'.format(each_delta))
        ax.set_xlabel('time steps')

    for i, each_delta in enumerate(delta_values):
        ax = axs_list[i][1]
        for mode in ['sq_oracle', 'lr_oracle', 'lin']:
            ax.plot(avg_reward_noquery[mode], label=modes[mode][0], color=modes[mode][1], marker=modes[mode][2], markevery=markerevery)
        for mode in ['mixI','mixII', 'mixIII']:
            ax.plot(avg_reward_noquery[mode][each_delta], label=modes[mode][0], color=modes[mode][1], marker=modes[mode][2], markevery=markerevery)
        ax.set_title('Average Autonomous Reward ($\Delta={}$)'.format(each_delta))
        ax.set_xlabel('time steps')

    handles, labels = axs_list[0][0].get_legend_handles_labels()
    for i, each_delta in enumerate(delta_values):
        fig = fig_list[i]
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.0),
                  fancybox=True, shadow=True, ncol=6)    
        fig.tight_layout()
        fig.savefig(os.path.join(Figure_dir, f'delta{each_delta}.pdf'), format='pdf', dpi=300, bbox_inches='tight')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for processing data.")
    parser.add_argument("--data_name", type=str, default='', help="Name of the dataset to use.")
    args = parser.parse_args()
    data_name = args.data_name

    result_root = data_name
    Figure_dir = f'Figures/{result_root}'
    os.makedirs(Figure_dir, exist_ok=True)
    plot_mixucbs(Figure_dir=Figure_dir, result_root=result_root, data_name=data_name)

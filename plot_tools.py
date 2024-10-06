import matplotlib.pyplot as plt
import os
import pickle
import numpy as np

def plot_average_rewards(axs, cumulative_rewards, cumulative_awards_std=None, params=None):
    """Plot average rewards in a 1 by m grid for different parameters."""
    for idx in range(len(axs)):
        ax = axs[idx]
        for key, item in cumulative_rewards.items():
            cumulative_reward = item[idx]
            # print('cumulative_reward', cumulative_reward)
            average_rewards = [cum_reward / (i + 1) for i, cum_reward in enumerate(cumulative_reward)]

            if cumulative_awards_std:
                # print(key, idx)
                # print(cumulative_awards_std[key])
                std = cumulative_awards_std[key][idx]
                average_rewards_std = [s / (i + 1) for i, s in enumerate(std)]

                ax.fill_between(range(len(average_rewards)),
                                [a - s for a, s in zip(average_rewards, average_rewards_std)],
                                [a + s for a, s in zip(average_rewards, average_rewards_std)], alpha=0.2)
            ax.plot(average_rewards, label=f'{key}')
            ax.set_xlabel('t')
            ax.set_ylabel('Average Reward')
            if params is not None:
                ax.set_title(f'$\\delta={params[idx]}$')
        ax.legend()

def plot_cumulative_rewards(axs, cumulative_rewards, cumulative_awards_std=None):
    """Plot average rewards in a 1 by m grid for different parameters."""
    for idx in range(len(axs)):
        ax = axs[idx]
        for key, item in cumulative_rewards.items():
            cumulative_reward = item[idx]
            if cumulative_awards_std:
                std = cumulative_awards_std[key][idx]
                ax.fill_between(range(len(cumulative_reward)), [a - s for a, s in zip(cumulative_reward, std)], [a + s for a, s in zip(cumulative_reward, std)], alpha=0.2)
            ax.plot(cumulative_reward, label=f'{key}')
            ax.set_xlabel('t')
            ax.set_ylabel('Cumulative Reward')
        ax.legend()

def plot_six_baselines(Figure_dir='Figures'):
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

    linucb_pkls = os.listdir('linucb_results')
    linucb_list = []
    for each_linucb_pkl in linucb_pkls:
        with open(os.path.join('linucb_results', each_linucb_pkl), 'rb') as f:
            data = pickle.load(f)
            CR_linucb = data['CR_linucb']
            linucb_list.append(CR_linucb)
    delta = 0.5
    mixucbI_pkls = os.listdir('mixucbI_results/{}'.format(delta))
    mixucbI_list = []
    for each_mixucbI_pkl in mixucbI_pkls:
        with open(os.path.join('mixucbI_results/{}'.format(delta), each_mixucbI_pkl), 'rb') as f:
            data = pickle.load(f)
            CR_mixucbI = data['CR_mixucbI']
            mixucbI_list.append(CR_mixucbI)

    mixucbII_pkls = os.listdir('mixucbII_results/{}'.format(delta))
    mixucbII_list = []
    for each_mixucbII_pkl in mixucbII_pkls:
        with open(os.path.join('mixucbII_results/{}'.format(delta), each_mixucbII_pkl), 'rb') as f:
            data = pickle.load(f)
            CR_mixucbII = data['CR_mixucbII']
            mixucbII_list.append(CR_mixucbII)

    mixucbIII_pkls = os.listdir('mixucbIII_results/{}'.format(delta))
    mixucbIII_list = []
    for each_mixucbIII_pkl in mixucbIII_pkls:
        with open(os.path.join('mixucbIII_results/{}'.format(delta), each_mixucbIII_pkl), 'rb') as f:
            data = pickle.load(f)
            CR_mixucbIII = data['CR_mixucbIII']
            mixucbIII_list.append(CR_mixucbIII)

    noisy_expert_pkls = os.listdir('noisy_expert_results')
    noisy_expert_list = []
    for each_noisy_expert_pkl in noisy_expert_pkls:
        with open(os.path.join('noisy_expert_results', each_noisy_expert_pkl), 'rb') as f:
            data = pickle.load(f)
            CR_NoisyExpert = data['CR_NoisyExpert']
            noisy_expert_list.append(CR_NoisyExpert)

    perfect_expert_pkls = os.listdir('perfect_expert_results')
    perfect_expert_list = []
    for each_perfect_expert_pkl in perfect_expert_pkls:
        with open(os.path.join('perfect_expert_results', each_perfect_expert_pkl), 'rb') as f:
            data = pickle.load(f)
            CR_PerfectExpert = data['CR_PerfectExpert']
            perfect_expert_list.append(CR_PerfectExpert)

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

    cumulative_rewards = {
        'LinUCB': CR_linucb_mean,
        'MixUCB-I ($\\delta={}$)'.format(delta): CR_mixucbI_mean,
        'MixUCB-II ($\\delta={}$)'.format(delta): CR_mixucbII_mean,
        'MixUCB-III ($\\delta={}$)'.format(delta): CR_mixucbIII_mean,
        'NoisyExpert': CR_NoisyExpert_mean,
        'PerfectExpert': CR_PerfectExpert_mean,
    }
    cumulative_rewards_std = {
        'LinUCB': CR_linucb_std,
        'MixUCB-I ($\\delta={}$)'.format(delta): CR_mixucbI_std,
        'MixUCB-II ($\\delta={}$)'.format(delta): CR_mixucbII_std,
        'MixUCB-III ($\\delta={}$)'.format(delta): CR_mixucbIII_std,
        'NoisyExpert': CR_NoisyExpert_std,
        'PerfectExpert': CR_PerfectExpert_std,
    }
    fig, axs = plt.subplots(1, 1, figsize=(8, 8))
    plot_average_rewards([axs], cumulative_rewards, cumulative_rewards_std)
    # plot_cumulative_rewards([axs], cumulative_rewards, cumulative_rewards_std)
    fig.savefig(os.path.join(Figure_dir, f'six_baselines.png'), format='jpg', dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

def plot_three_mixucbs(Figure_dir='Figures'):
    os.makedirs(Figure_dir, exist_ok=True)
    CR_mixucbI_mean = []
    CR_mixucbI_std = []
    CR_mixucbII_mean = []
    CR_mixucbII_std = []
    CR_mixucbIII_mean = []
    CR_mixucbIII_std = []
    delta_values = [0.2, 0.5, 1.,2., 5.]
    for each_delta in delta_values:
        mixucbI_pkls = os.listdir(os.path.join('mixucbI_results','{}'.format(each_delta)))
        mixucbI_list = []
        for each_mixucbI_pkl in mixucbI_pkls:
            with open(os.path.join('mixucbI_results','{}'.format(each_delta), each_mixucbI_pkl), 'rb') as f:
                data = pickle.load(f)
                CR_mixucbI = data['CR_mixucbI']
                mixucbI_list.append(CR_mixucbI)

        mixucbII_pkls = os.listdir(os.path.join('mixucbII_results','{}'.format(each_delta)))
        mixucbII_list = []
        for each_mixucbII_pkl in mixucbII_pkls:
            with open(os.path.join('mixucbII_results','{}'.format(each_delta), each_mixucbII_pkl), 'rb') as f:
                data = pickle.load(f)
                CR_mixucbII = data['CR_mixucbII']
                mixucbII_list.append(CR_mixucbII)

        mixucbIII_pkls = os.listdir(os.path.join('mixucbIII_results','{}'.format(each_delta)))
        mixucbIII_list = []
        for each_mixucbIII_pkl in mixucbIII_pkls:
            with open(os.path.join('mixucbIII_results','{}'.format(each_delta), each_mixucbIII_pkl), 'rb') as f:
                data = pickle.load(f)
                CR_mixucbIII = data['CR_mixucbIII']
                mixucbIII_list.append(CR_mixucbIII)

        CR_mixucbI_mean.append(np.mean(mixucbI_list, axis=0))
        CR_mixucbI_std.append(np.std(mixucbI_list, axis=0))
        CR_mixucbII_mean.append(np.mean(mixucbII_list, axis=0))
        CR_mixucbII_std.append(np.std(mixucbII_list, axis=0))
        CR_mixucbIII_mean.append(np.mean(mixucbIII_list, axis=0))
        CR_mixucbIII_std.append(np.std(mixucbIII_list, axis=0))

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
    # Create a 1 by m grid of subplots for average rewards
    fig, axs = plt.subplots(1, len(delta_values), figsize=(18, 3))

    plot_average_rewards(axs, cumulative_rewards, cumulative_rewards_std, delta_values)
    # plot_cumulative_rewards([axs], cumulative_rewards, cumulative_rewards_std)
    plt.tight_layout()
    fig.savefig(os.path.join(Figure_dir, f'three_mixucbs.png'), format='jpg', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    plot_three_mixucbs()
    plot_six_baselines()
import matplotlib.pyplot as plt
import os
import pickle
import numpy as np

def plot_average_rewards(axs, cumulative_rewards, params, cumulative_awards_std=None):
    """Plot average rewards in a 1 by m grid for different parameters."""
    for idx in range(len(axs)):
        ax = axs[idx]
        for key, item in cumulative_rewards.items():
            cumulative_reward = item[idx]
            # print('cumulative_reward', cumulative_reward)
            average_rewards = [cum_reward / (i + 1) for i, cum_reward in enumerate(cumulative_reward)]

            if cumulative_awards_std:
                std = cumulative_awards_std[key][idx]
                average_rewards_std = [s / (i + 1) for i, s in enumerate(std)]

                ax.fill_between(range(len(average_rewards)),
                                [a - s for a, s in zip(average_rewards, average_rewards_std)],
                                [a + s for a, s in zip(average_rewards, average_rewards_std)], alpha=0.2)
            ax.plot(average_rewards, label=f'{key}')
            ax.set_xlabel('t')
            ax.set_ylabel('Average Reward')
            # ax.set_title(f'$\\alpha={params[idx]}$')
        ax.legend()

def plot_six_baselines(Figure_dir='Figures'):
    # os.makedirs(Figure_dir, exist_ok=True)
    linucb_pkls = os.listdir('linucb_results')
    with open(os.path.join('linucb_results', linucb_pkls[0]), 'rb') as f:
        data = pickle.load(f)
        CR_linucb = data['CR_linucb']

    mixucbI_pkls = os.listdir('mixucbI_results')
    with open(os.path.join('mixucbI_results', mixucbI_pkls[0]), 'rb') as f:
        data = pickle.load(f)
        CR_mixucbI = data['CR_mixucbI']

    mixucbII_pkls = os.listdir('mixucbII_results')
    with open(os.path.join('mixucbII_results', mixucbII_pkls[0]), 'rb') as f:
        data = pickle.load(f)
        CR_mixucbII = data['CR_mixucbII']

    mixucbIII_pkls = os.listdir('mixucbIII_results')
    with open(os.path.join('mixucbIII_results', mixucbIII_pkls[0]), 'rb') as f:
        data = pickle.load(f)
        CR_mixucbIII = data['CR_mixucbIII']

    noisy_expert_pkls = os.listdir('noisy_expert_results')
    with open(os.path.join('noisy_expert_results', noisy_expert_pkls[0]), 'rb') as f:
        data = pickle.load(f)
        CR_NoisyExpert = data['CR_NoisyExpert']

    perfect_expert_pkls = os.listdir('perfect_expert_results')
    with open(os.path.join('perfect_expert_results', perfect_expert_pkls[0]), 'rb') as f:
        data = pickle.load(f)
        CR_PerfectExpert = data['CR_PerfectExpert']

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

    print(len(CR_linucb))
    # CR_linucb_mean.append(np.mean(CR_linucb, axis=0))
    # CR_linucb_std.append(np.std(CR_linucb, axis=0))
    # CR_mixucbI_mean.append(np.mean(CR_mixucbI, axis=0))
    # CR_mixucbI_std.append(np.std(CR_mixucbI, axis=0))
    # CR_mixucbII_mean.append(np.mean(CR_mixucbII, axis=0))
    # CR_mixucbII_std.append(np.std(CR_mixucbII, axis=0))
    # CR_mixucbIII_mean.append(np.mean(CR_mixucbIII, axis=0))
    # CR_mixucbIII_std.append(np.std(CR_mixucbIII, axis=0))
    # CR_NoisyExpert_mean.append(np.mean(CR_NoisyExpert, axis=0))
    # CR_NoisyExpert_std.append(np.std(CR_NoisyExpert, axis=0))
    # CR_PerfectExpert_mean.append(np.mean(CR_PerfectExpert, axis=0))
    # CR_PerfectExpert_std.append(np.std(CR_PerfectExpert, axis=0))

    CR_linucb_mean.append(CR_linucb)
    CR_linucb_std.append(CR_linucb)
    CR_mixucbI_mean.append(CR_mixucbI)
    CR_mixucbI_std.append(CR_mixucbI)
    CR_mixucbII_mean.append(CR_mixucbII)
    CR_mixucbII_std.append(CR_mixucbII)
    CR_mixucbIII_mean.append(CR_mixucbIII)
    CR_mixucbIII_std.append(CR_mixucbIII)
    CR_NoisyExpert_mean.append(CR_NoisyExpert)
    CR_NoisyExpert_std.append(CR_NoisyExpert)
    CR_PerfectExpert_mean.append(CR_PerfectExpert)
    CR_PerfectExpert_std.append(CR_PerfectExpert)

    cumulative_rewards = {
        'LinUCB': CR_linucb_mean,
        'MixUCB-I': CR_mixucbI_mean,
        'MixUCB-II': CR_mixucbII_mean,
        'MixUCB-III': CR_mixucbIII_mean,
        'NoisyExpert': CR_NoisyExpert_mean,
        'PerfectExpert': CR_PerfectExpert_mean,
    }
    cumulative_rewards_std = {
        'LinUCB': CR_linucb_std,
        'MixUCB-I': CR_mixucbI_std,
        'MixUCB-II': CR_mixucbII_std,
        'MixUCB-III': CR_mixucbIII_std,
        'NoisyExpert': CR_NoisyExpert_std,
        'PerfectExpert': CR_PerfectExpert_std,
    }
    fig, axs = plt.subplots(1, 1, figsize=(8, 8))
    plot_average_rewards([axs], cumulative_rewards, cumulative_rewards_std)
    fig.savefig(os.path.join(Figure_dir, f'six_baselines.jpg'), format='jpg', dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_six_baselines()
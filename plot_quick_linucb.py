## Quick and dirty plotting script to compare LinUCB to perfect expert.
## we should see sublinear regret for LinUCB.

import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

from plot_tools import plot_cumulative_rewards, plot_average_rewards

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--setting_id', type=int, default=0, help='Setting ID for the experiment')

def main(args):
    result_root = ''
    # result_root = 'g2temp1.0_linearreward_20241008'
    Figure_dir=f'Figures/linucb_{args.setting_id}'
    os.makedirs(Figure_dir, exist_ok=True)

    CR_linucb_mean = []
    CR_linucb_std = []
    CR_PerfectExpert_mean = []
    CR_PerfectExpert_std = []

    RRlinucb_mean = []
    RRlinucb_std = []

    # IR = instantanous regret
    IR_linucb_mean = []
    IR_linucb_std = []

    perfect_expert_pkls = os.listdir(os.path.join(result_root,f'perfect_expert_results_{args.setting_id}'))
    perfect_expert_list = []
    perfect_expert_rawreward_list = []
    for each_perfect_expert_pkl in perfect_expert_pkls:
        with open(os.path.join(result_root,f'perfect_expert_results_{args.setting_id}', each_perfect_expert_pkl), 'rb') as f:
            data = pkl.load(f)
            CR_PerfectExpert = data['CR_PerfectExpert']
            perfect_expert_list.append(CR_PerfectExpert)
            # extract raw rewards from cumulative reward list.
            raw_rewards = [CR_PerfectExpert[i] - CR_PerfectExpert[i-1] if i > 0 else CR_PerfectExpert[i] for i in range(len(CR_PerfectExpert))]
            perfect_expert_rawreward_list.append(raw_rewards)

    linucb_pkls = os.listdir(os.path.join(result_root,f'linucb_results_{args.setting_id}'))
    linucb_list = []
    linucb_rawreward_list = []
    for each_linucb_pkl in linucb_pkls:
        with open(os.path.join(result_root,f'linucb_results_{args.setting_id}', each_linucb_pkl), 'rb') as f:
            data = pkl.load(f)
            CR_linucb = data['CR_linucb']
            linucb_list.append(CR_linucb)
            # extract raw rewards from cumulative reward list.
            raw_rewards = [CR_linucb[i] - CR_linucb[i-1] if i > 0 else CR_linucb[i] for i in range(len(CR_linucb))]
            linucb_rawreward_list.append(raw_rewards)

    # assume that perfect expert is deterministic.
    RRlinucb_list = [np.array(perfect_expert_list[0]) - np.array(l) for l in linucb_list]
    IRlinucb_list = [np.array(perfect_expert_rawreward_list[0]) - np.array(l) for l in linucb_rawreward_list]
    
    
    RRlinucb_mean.append(np.mean(RRlinucb_list, axis=0))
    RRlinucb_std.append(np.std(RRlinucb_list, axis=0))

    IR_linucb_mean.append(np.mean(IRlinucb_list, axis=0))
    IR_linucb_std.append(np.std(IRlinucb_list, axis=0))

    CR_linucb_mean.append(np.mean(linucb_list, axis=0))
    CR_linucb_std.append(np.std(linucb_list, axis=0))
    CR_PerfectExpert_mean.append(np.mean(perfect_expert_list, axis=0))
    CR_PerfectExpert_std.append(np.std(perfect_expert_list, axis=0))

    cumulative_rewards = {
        'LinUCB': CR_linucb_mean,
        'PerfectExpert': CR_PerfectExpert_mean,
    }
    cumulative_rewards_std = {
    'LinUCB': CR_linucb_std,
        'PerfectExpert': CR_PerfectExpert_std,
    }

    rr = {
        'LinUCB': RRlinucb_mean,
    }

    rr_std = {
        'LinUCB': RRlinucb_std,
    }

    ir = {
        'LinUCB': IR_linucb_mean,
    }

    ir_std = {
        'LinUCB': IR_linucb_std,
    }


    fig, axs = plt.subplots(1, 1, figsize=(8, 8))
    plot_average_rewards([axs], cumulative_rewards, cumulative_rewards_std)
    fig.savefig(os.path.join(Figure_dir, f'linucb_avgr.png'), format='jpg', dpi=300, bbox_inches='tight')

    fig, axs = plt.subplots(1, 1, figsize=(8, 8))
    plot_cumulative_rewards([axs], cumulative_rewards, cumulative_rewards_std)
    fig.savefig(os.path.join(Figure_dir, f'linucb_cr.png'), format='jpg', dpi=300, bbox_inches='tight')
    plt.tight_layout()

    # Add a reward regret plot, which is difference between perfect expert and querying algorithm.
    fig, axs = plt.subplots(1, 1, figsize=(8, 8))
    plot_cumulative_rewards([axs], rr, rr_std,ylabel="Reward Regret")
    plt.tight_layout()
    fig.savefig(os.path.join(Figure_dir, f'linucb_rr.png'), format='jpg', dpi=300, bbox_inches='tight')

    # add an average rr plot.
    fig, axs = plt.subplots(1, 1, figsize=(8, 8))
    plot_average_rewards([axs], rr, rr_std, ylabel="Average Reward Regret")
    plt.tight_layout()
    fig.savefig(os.path.join(Figure_dir, f'linucb_avgrr.png'), format='jpg', dpi=300, bbox_inches='tight')

    # add an instantaneous regret plot.
    fig, axs = plt.subplots(1, 1, figsize=(8, 8))
    plot_cumulative_rewards([axs], ir, ir_std, ylabel="Instantaneous Regret")
    plt.tight_layout()
    fig.savefig(os.path.join(Figure_dir, f'linucb_ir.png'), format='jpg', dpi=300, bbox_inches='tight')

if __name__=="__main__":
    args = parser.parse_args()
    main(args)

from matplotlib.patches import Ellipse
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv
# Function to plot ellipses
def plot_ellipses(ax, covariance_matrix, theta, color='blue', label=None, hatch=None, alpha=0.2):
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width, height = 2 * np.sqrt(eigenvalues)
    
    ellipse = Ellipse(xy=theta, width=width, height=height, angle=angle, edgecolor=color, facecolor='none', alpha=alpha, hatch=hatch, label=label)
    ax.add_patch(ellipse)
    
    

def plot_average_rewards(axs, cumulative_rewards, params, cumulative_awards_std=None):
    """Plot average rewards in a 1 by m grid for different parameters."""
    for idx in range(len(params)):
        ax = axs[idx]
        for key, item in cumulative_rewards.items():
            cumulative_reward = item[idx]
            average_rewards = [cum_reward / (i + 1) for i, cum_reward in enumerate(cumulative_reward)]
            
            if cumulative_awards_std:
                std = cumulative_awards_std[key][idx]
                average_rewards_std = [s / (i + 1) for i, s in enumerate(std)]
                
                ax.fill_between(range(len(average_rewards)), [a - s for a, s in zip(average_rewards, average_rewards_std)], [a + s for a, s in zip(average_rewards, average_rewards_std)], alpha=0.2)
            ax.plot(average_rewards, label=f'{key}')
            ax.set_xlabel('t')
            ax.set_ylabel('Average Reward')
            ax.set_title(f'$\\alpha={params[idx]}$')
        ax.legend()
def plot_cumulative_rewards(axs, cumulative_rewards, params, cumulative_awards_std=None):
    """Plot average rewards in a 1 by m grid for different parameters."""
    for idx in range(len(params)):
        ax = axs[idx]
        for key, item in cumulative_rewards.items():
            cumulative_reward = item[idx]
            if cumulative_awards_std:
                std = cumulative_awards_std[key][idx]
                ax.fill_between(range(len(cumulative_reward)), [a - s for a, s in zip(cumulative_reward, std)], [a + s for a, s in zip(cumulative_reward, std)], alpha=0.2)
            ax.plot(cumulative_reward, label=f'{key}')
            ax.set_xlabel('t')
            ax.set_ylabel('Cumulative Reward')
            ax.set_title(f'$\\alpha={params[idx]}$')
        ax.legend()  
    
    
def plot_query_timelines(axs, query_data, params):
    """Plot query timelines in a 1 by m grid for different parameters."""
    for idx, (q, param) in enumerate(zip(query_data, params)):
        ax = axs[idx]
        ax.bar(range(len(q)), q, color='blue', alpha=0.6)
        ax.set_xlabel('t')
        ax.set_title(f'$\\alpha={param}$')
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['No Query', 'Query'])
        
        
        






def plot_ellipses_grid(axs, plot_rounds, params, theta_data, cov_data, colors):
    """Plot ellipses on a grid of subplots for different rounds and parameters."""
    xlim, ylim = [np.inf, -np.inf], [np.inf, -np.inf]


    for col_idx in range(len(plot_rounds)):
        round_num = plot_rounds[col_idx]
        ax = axs[col_idx]
        linucb_theta = theta_data[round_num]['linucb']
        mixucb_theta = theta_data[round_num]['mixucb']
        always_query_theta = theta_data[round_num]['always_query_ucb']
        covariance_matrix_linucb = cov_data[round_num]['linucb']
        covariance_matrix_mixucb = cov_data[round_num]['mixucb']
        covariance_matrix_always_query = cov_data[round_num]['always_query_ucb']

        
        
        
        plot_ellipses(ax, covariance_matrix_linucb, linucb_theta, color= colors['linucb'], label='LinUCB', hatch='----', alpha=0.5)
        plot_ellipses(ax, covariance_matrix_mixucb, mixucb_theta, color= colors['mixucb'], label='MixUCB', hatch='\\\\\\\\', alpha=0.5)
        plot_ellipses(ax, covariance_matrix_always_query, always_query_theta, color= colors['always_query'], hatch='////',label='Always Query', alpha=0.5)


        
        ax.relim()
        ax.autoscale_view()
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f't= {round_num}')
        if col_idx == 0:
            ax.legend()

        current_xlim = ax.get_xlim()
        current_ylim = ax.get_ylim()
        xlim = [min(xlim[0], current_xlim[0]), max(xlim[1], current_xlim[1])]
        ylim = [min(ylim[0], current_ylim[0]), max(ylim[1], current_ylim[1])]

    largest_limit = max(xlim[1] - xlim[0], ylim[1] - ylim[0])
    mid_x = (xlim[0] + xlim[1]) / 2
    mid_y = (ylim[0] + ylim[1]) / 2

    equal_xlim = [mid_x - largest_limit / 2, mid_x + largest_limit / 2]
    equal_ylim = [mid_y - largest_limit / 2, mid_y + largest_limit / 2]

    # Second loop to apply the equalized limits to all subplots

    for col_idx in range(len(plot_rounds)):
        ax = axs[col_idx]
        ax.set_xlim(equal_xlim)
        ax.set_ylim(equal_ylim)
        





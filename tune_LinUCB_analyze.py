## Analyze results of LinUCB tuning experiment.
import itertools
import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Parameters to tune: alpha and lambda
    alphas = [0.1, 0.5, 1, 10]
    lambdas = [0.001, 0.01, 0.1, 1, 2, 4, 6, 8, 10]
    generator = list(itertools.product(alphas, lambdas))
    
    final_CRs = [[0 for _ in range(len(lambdas))] for _ in range(len(alphas))]

    CR_linucb_mean = []     # len of list = # of hyperparameter settings
    CR_linucb_std = []

    for (setting_id, (alpha, lambda_)) in enumerate(generator):
        result_root=''
        linucb_pkls = os.listdir(os.path.join(f'linucb_results_{setting_id}'))
        linucb_list = []
        for each_linucb_pkl in linucb_pkls:
            with open(os.path.join(result_root,f'linucb_results_{setting_id}', each_linucb_pkl), 'rb') as f:
                data = pkl.load(f)
                CR_linucb = data['CR_linucb']
                linucb_list.append(CR_linucb)

        CR_linucb_mean.append(np.mean(linucb_list, axis=0))
        CR_linucb_std.append(np.std(linucb_list, axis=0))

        # Log just the final cumulative reward for now
        final_CRs[alphas.index(alpha)][lambdas.index(lambda_)] = CR_linucb_mean[-1][-1]

    print(final_CRs)

    # Find the argmax setting.
    array_vers = np.array(final_CRs)
    index = np.argmax(array_vers)
    alpha_index = index // len(lambdas)
    lambda_index = index % len(lambdas)
    print(f"Best setting: alpha={alphas[alpha_index]}, lambda={lambdas[lambda_index]}")

    # Generate heatmap of final cumulative rewards
    fig, ax = plt.subplots()
    im = ax.imshow(final_CRs)
    ax.set_title('Heatmap')
    ax.set_xlabel('lambda')
    ax.set_ylabel('alpha')
    ax.set_xticks(range(len(lambdas)), lambdas)
    ax.set_yticks(range(len(alphas)), alphas)
    # add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    plt.savefig('Figures/linucb_heatmap.png')


if __name__=="__main__":
    main()
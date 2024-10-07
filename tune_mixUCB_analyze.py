### Analyze result from tune_mixUCB.py

## Things to check:
## (1) what do the failed pickles look like?
## (2) what does the output of the run_mixucbI.py, run_mixucbII.py, and run_mixucbIII.py look like,
##     specifically, what is the number of queries made by each algorithm?

import pickle as pkl
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import os
import numpy as np
import sys

def main():
    # (1) load the failed pickles
    failed_I = pkl.load(open('failed_I.pkl', 'rb'))
    failed_II = pkl.load(open('failed_II.pkl', 'rb'))
    print(f"Failure setting for I: {failed_I}")
    print(f"Failure setting for II: {failed_II}")

    # lambdas = [0.001, 0.01, 0.1, 1]
    # beta_MixUCBI_values = [1000, 2000, 4000, 8000, 16000]

    # (lambda, beta) values
    # generator = [(1,5000),(1,6000),(1,7000)] + [(10,beta) for beta in [1000, 2000, 4000, 8000, 16000]]

    # 2024-10-6: experiment III
    lambdas = [2, 4, 6, 8]
    beta_MixUCBI_values = [1000, 2000, 4000, 8000]
    generator = [(lambda_, beta) for lambda_ in lambdas for beta in beta_MixUCBI_values]

    # Let's make binary heatmaps to see which settings failed
    failed_I_matrix = [[int(failed_I[(lambda_, beta)]) for beta in beta_MixUCBI_values] for lambda_ in lambdas]
    failed_II_matrix = [[int(failed_II[(lambda_, beta)]) for beta in beta_MixUCBI_values] for lambda_ in lambdas]
    
    fig, ax = plt.subplots()
    ax.imshow(failed_I_matrix, cmap='binary')
    ax.set_title('MixUCB-I (black = failed, white = succeeded)')
    ax.set_xlabel('beta')
    ax.set_ylabel('lambda')
    ax.set_xticks(range(len(beta_MixUCBI_values)), beta_MixUCBI_values)
    ax.set_yticks(range(len(lambdas)), lambdas)
    plt.savefig('Figures/failed_I.png')

    fig, ax = plt.subplots()
    ax.imshow(failed_II_matrix, cmap='binary')
    ax.set_title('MixUCB-II (black = failed, white = succeeded)')
    ax.set_xlabel('beta')
    ax.set_ylabel('lambda')
    ax.set_xticks(range(len(beta_MixUCBI_values)), beta_MixUCBI_values)
    ax.set_yticks(range(len(lambdas)), lambdas)
    plt.savefig('Figures/failed_II.png')


    # (2) Let's examine the number of queries specifically for the following setting
    # (lambda, beta) = (1, 1000)
    # Actually, let's go through a few settings.

    # Initial setting
    # settings = [(1,8000),(1,16000),(0.1,16000),(0.01,16000)]

    # Better logic:
    # Instead, go through all settings for which both MixUCB-I and MixUCB-II succeeded.
    settings = [pair for pair in generator if (not failed_I[pair] and not failed_II[pair])]

    print(f"Common settings where both I and II didn't fail to converge: {settings}")

    for (lambda_to_check, beta_to_check) in settings:
        print(f"Checking querying metrics for lambda={lambda_to_check} and beta={beta_to_check}")
        # setting_ID = lambdas.index(lambda_to_check) * len(beta_MixUCBI_values) + beta_MixUCBI_values.index(beta_to_check)
        setting_ID = generator.index((lambda_to_check, beta_to_check))
        print(f"Setting ID: {setting_ID}")

        CR_mixucbI_mean = []
        CR_mixucbI_std = []
        CR_mixucbII_mean = []
        CR_mixucbII_std = []
        CR_mixucbIII_mean = []
        CR_mixucbIII_std = []

        TotalQ_mixucbI_mean = []
        TotalQ_mixucbI_std = []
        TotalQ_mixucbII_mean = []
        TotalQ_mixucbII_std = []
        TotalQ_mixUCBIII_mean = []
        TotalQ_mixUCBIII_std = []

        delta_values = [0.2, 0.5, 1.,2., 5.]
        for each_delta in delta_values:
            mixucbI_pkls = os.listdir(os.path.join(f'mixucbI_results_{setting_ID}','{}'.format(each_delta)))
            mixucbI_list = []
            mixucbI_list_totalQ = []
            for each_mixucbI_pkl in mixucbI_pkls:
                with open(os.path.join(f'mixucbI_results_{setting_ID}','{}'.format(each_delta), each_mixucbI_pkl), 'rb') as f:
                    data = pkl.load(f)
                    CR_mixucbI = data['CR_mixucbI']
                    mixucbI_list.append(CR_mixucbI)
                    TotalQ_mixUCBI = data['TotalQ_mixucbI']
                    mixucbI_list_totalQ.append(TotalQ_mixUCBI)

            mixucbII_pkls = os.listdir(os.path.join(f'mixucbII_results_{setting_ID}','{}'.format(each_delta)))
            mixucbII_list = []
            mixucbII_list_totalQ = []
            for each_mixucbII_pkl in mixucbII_pkls:
                with open(os.path.join(f'mixucbII_results_{setting_ID}','{}'.format(each_delta), each_mixucbII_pkl), 'rb') as f:
                    data = pkl.load(f)
                    CR_mixucbII = data['CR_mixucbII']
                    mixucbII_list.append(CR_mixucbII)
                    TotalQ_mixUCBII = data['TotalQ_mixucbII']
                    mixucbII_list_totalQ.append(TotalQ_mixUCBII)

            mixucbIII_pkls = os.listdir(os.path.join(f'mixucbIII_results_{setting_ID}','{}'.format(each_delta)))
            mixucbIII_list = []
            mixucbIII_list_totalQ = []
            for each_mixucbIII_pkl in mixucbIII_pkls:
                with open(os.path.join(f'mixucbIII_results_{setting_ID}','{}'.format(each_delta), each_mixucbIII_pkl), 'rb') as f:
                    data = pkl.load(f)
                    CR_mixucbIII = data['CR_mixucbIII']
                    mixucbIII_list.append(CR_mixucbIII)
                    TotalQ_mixUCBIII = data['TotalQ_mixucbIII']
                    mixucbIII_list_totalQ.append(TotalQ_mixUCBIII)

            CR_mixucbI_mean.append(np.mean(mixucbI_list, axis=0))
            CR_mixucbI_std.append(np.std(mixucbI_list, axis=0))
            CR_mixucbII_mean.append(np.mean(mixucbII_list, axis=0))
            CR_mixucbII_std.append(np.std(mixucbII_list, axis=0))
            CR_mixucbIII_mean.append(np.mean(mixucbIII_list, axis=0))
            CR_mixucbIII_std.append(np.std(mixucbIII_list, axis=0))

            TotalQ_mixucbI_mean.append(np.mean(mixucbI_list_totalQ))
            TotalQ_mixucbI_std.append(np.std(mixucbI_list_totalQ))
            TotalQ_mixucbII_mean.append(np.mean(mixucbII_list_totalQ))
            TotalQ_mixucbII_std.append(np.std(mixucbII_list_totalQ))
            TotalQ_mixUCBIII_mean.append(np.mean(mixucbIII_list_totalQ))
            TotalQ_mixUCBIII_std.append(np.std(mixucbIII_list_totalQ))

        # Print querying metrics for now. Just means.
        print("Querying metrics for MixUCB-I")
        print(f"Mean of TotalQ_mixucbI: {TotalQ_mixucbI_mean}")
        print(f"Mean of TotalQ_mixucbII: {TotalQ_mixucbII_mean}")
        print(f"Mean of TotalQ_mixUCBIII: {TotalQ_mixUCBIII_mean}")

if __name__=="__main__":
    main()
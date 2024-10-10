## Tuning script for mixUCBI (for SPANET data)
## To find a setting of betaMixUCBI that leads to convergence for all runs.

import run_mixucbI
import run_mixucbII
import run_mixucbIII
import cvxpy
import numpy as np
import pickle as pkl

import itertools
from argparse import ArgumentParser

# beta_MixUCBI_values = [9000,10000,11000,12000,13000,14000,15000,16000,17000,18000,19000,20000]

# 10/9: sticking to higher betas for now, for speed (so we can finish this in 1 hr).
# beta_MixUCBI_values = [15000,16000,17000,18000,19000,20000]

# 10/9: synthetic experiments with noise_std=0.2
data_file = "simulation_data_toy20241009_noise0.2.pkl"
# beta_MixUCBI_values = [1000, 2000, 4000, 8000]
# higher betas
# beta_MixUCBI_values = [8000, 10000, 12000, 14000, 16000]
# all the betas
# beta_MixUCBI_values = [1000, 2000, 4000, 8000, 10000, 12000, 14000, 16000]
# what about smaller betas?
# beta_MixUCBI_values = [100,200,300,400,500]
# back to original range
# beta_MixUCBI_values = [1000, 2000, 4000, 8000, 16000]
beta_MixUCBI_values = [1000, 2000, 4000, 8000, 12000]
# alpha = 0.1
# lambda_ = 0.001
# lambda_ = 1
# I think we need to do a lambda sweep as well.
lambdas = [0.001, 0.01, 0.1, 1]
generator = list(itertools.product(lambdas, beta_MixUCBI_values))
T = 200
deltas=[0.2,0.5,1.0]

def main(temperature):
    # Fixed parameters: for experiments in 10/6 and before.
    # temperature = 0.1
    # alpha = 1
    # Fixed parameters: for experiments from 10/7.
    # temperature = 10
    # alpha = 1
    # Fixed parameters for experiments on night of 10/7.
    # temperature is also considered to be fixed.
    # alpha = 0.1
    # lambda_ = 1
    # print(f"Running all algorithms with temperature={temperature}, alpha={alpha}, lambda={lambda_}...")
    print(f"Running all algorithms with temperature={temperature}, alpha={alpha}...")

    # Variable parameters.
    # lambdas = [0.001, 0.01, 0.1, 1]
    # beta_MixUCBI_values = [1000, 2000, 4000, 8000, 16000]
    # generator = itertools.product(lambdas, beta_MixUCBI_values)

    # 10/6 - second experiment.
    # generator = [(1,5000),(1,6000),(1,7000)] + [(10,beta) for beta in [1000, 2000, 4000, 8000, 16000]]

    # 10/6 - third experiments
    # lambdas = [2, 4, 6, 8]
    # beta_MixUCBI_values = [1000, 2000, 4000, 8000]
    # generator = list(itertools.product(lambdas, beta_MixUCBI_values))

    # 10/7 - round 1 (higher temperature expert = less noisy)
    # lambdas = [0.01, 0.1, 1, 2, 4, 6]
    # beta_MixUCBI_values = [1000, 2000, 4000, 8000]
    # generator = list(itertools.product(lambdas, beta_MixUCBI_values))

    # 10/7 - fixing temp, alpha, lambda
    # beta_MixUCBI_values = [5000, 6000, 7000, 8000]

    # 10/7: higher range of beta_MixUCBI
    # 10/8: using linear reward oracle.

    # failed_I = {(lambda_, beta): False for (lambda_, beta) in generator}
    # failed_II = {(lambda_, beta): False for (lambda_, beta) in generator}
    failed_I = {beta: False for beta in beta_MixUCBI_values}
    failed_II = {beta: False for beta in beta_MixUCBI_values}

    for (setting_id, (lambda_, beta)) in enumerate(generator):
    # for (setting_id, beta) in enumerate(beta_MixUCBI_values):
        # print(f"Running all algorithms with lambda={lambda_} and beta_MixUCBI={beta}...")
        print(f"Running all algorithms with beta={beta}...")
        # MixUCB-I
        args = run_mixucbI.parser.parse_args(['--pickle_file', data_file, '--temperature', str(temperature), \
                                              '--alpha', str(alpha), \
                                              '--beta_MixUCBI', str(beta), '--lambda_', str(lambda_), '--setting_id', f"temp{temperature}_{setting_id}"])
        try:
            run_mixucbI.main(args)
        except cvxpy.error.SolverError as e:
            failed_I[beta] = True
        except cvxpy.error.ParameterError as e:
            print(e)
            import pdb; pdb.set_trace()
        # MixUCB-II
        args = run_mixucbII.parser.parse_args(['--pickle_file', data_file, '--temperature', str(temperature), \
                                              '--alpha', str(alpha), \
                                              '--beta_MixUCBII', str(beta), '--lambda_', str(lambda_), '--setting_id', f"temp{temperature}_{setting_id}"])
        try:
            run_mixucbII.main(args)
        except cvxpy.error.SolverError as e:
            failed_II[beta] = True
        except cvxpy.error.ParameterError as e:
            print(e)
            import pdb; pdb.set_trace()
        # MixUCB-III
        args = run_mixucbIII.parser.parse_args(['--pickle_file', data_file, \
                                              '--alpha', str(alpha), \
                                              '--lambda_', str(lambda_), '--setting_id', f"temp{temperature}_{setting_id}"])
        run_mixucbIII.main(args)
            
    # pkl.dump(failed_I, open('failed_I.pkl', 'wb'))
    # pkl.dump(failed_II, open('failed_II.pkl', 'wb'))
    pkl.dump(failed_I, open(f'failed_I_{temperature}.pkl', 'wb'))
    pkl.dump(failed_II, open(f'failed_II_{temperature}.pkl', 'wb'))

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument('--temperature', type=float, required=True)
    args = parser.parse_args()

    temperature = args.temperature
    main(temperature)
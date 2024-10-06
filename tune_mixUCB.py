## Tuning script for mixUCBI (for SPANET data)
## To find a setting of betaMixUCBI that leads to convergence for all runs.

import run_mixucbI
import run_mixucbII
import run_mixucbIII
import cvxpy
import numpy as np
import pickle as pkl

import itertools

if __name__=="__main__":
    # Fixed parameters
    temperature = 0.1
    alpha = 1

    # Variable parameters.
    lambdas = [0.001, 0.01, 0.1, 1]
    beta_MixUCBI_values = [1000, 2000, 4000, 8000, 16000]
    failed_I = {(lambda_, beta): False for (lambda_, beta) in itertools.product(lambdas, beta_MixUCBI_values)}
    failed_II = {(lambda_, beta): False for (lambda_, beta) in itertools.product(lambdas, beta_MixUCBI_values)}

    for (setting_id, (lambda_, beta)) in enumerate(itertools.product(lambdas, beta_MixUCBI_values)):
        print(f"Running all algorithms with lambda={lambda_} and beta_MixUCBI={beta}...")
        # MixUCB-I
        args = run_mixucbI.parser.parse_args(['--pickle_file', 'simulation_data_spanet.pkl', '--temperature', str(temperature), \
                                              '--alpha', str(alpha), \
                                              '--beta_MixUCBI', str(beta), '--lambda_', str(lambda_), '--setting_id', str(setting_id)])
        try:
            run_mixucbI.main(args)
        except cvxpy.error.SolverError as e:
            failed_I[(lambda_, beta)] = True
        # MixUCB-II
        args = run_mixucbII.parser.parse_args(['--pickle_file', 'simulation_data_spanet.pkl', '--temperature', str(temperature), \
                                              '--alpha', str(alpha), \
                                              '--beta_MixUCBII', str(beta), '--lambda_', str(lambda_), '--setting_id', str(setting_id)])
        try:
            run_mixucbII.main(args)
        except cvxpy.error.SolverError as e:
            failed_II[(lambda_, beta)] = True
        # MixUCB-III
        args = run_mixucbIII.parser.parse_args(['--pickle_file', 'simulation_data_spanet.pkl', \
                                              '--alpha', str(alpha), \
                                              '--lambda_', str(lambda_), '--setting_id', str(setting_id)])
        run_mixucbIII.main(args)
            
    pkl.dump(failed_I, open('failed_I.pkl', 'wb'))
    pkl.dump(failed_II, open('failed_II.pkl', 'wb'))

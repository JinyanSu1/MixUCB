## Goal here is to tune LinUCB hyperparameters, 
## specifically alpha and lambda.
## Performance metric for now could just be the cumulative reward
## at the end of T timesteps.

## Rough sketch
## - leverage run_linucb.py
## - loop over alpha and lambda values
## - generate a heatmap with the cumulative reward at the end of T timesteps.

import run_linucb
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
    
    for (setting_id, (alpha, lambda_)) in enumerate(generator):
        print(f"Running LinUCB with alpha={alpha} and lambda={lambda_}...")
        args = run_linucb.parser.parse_args(['--pickle_file', 'simulation_data_spanet.pkl', \
        '--T', '1000', \
        '--alpha', str(alpha), \
        '--lambda_', str(lambda_),\
        '--setting_id', str(setting_id)])
        run_linucb.main(args)

if __name__=="__main__":
    main()
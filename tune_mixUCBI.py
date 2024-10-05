## Tuning script for mixUCBI (for SPANET data)
## To find a setting of betaMixUCBI that leads to convergence for all runs.

import run_mixucbI
import cvxpy
import numpy as np

if __name__=="__main__":
    # beta_MixUCBI_values = [1000, 1500, 2000, 2500, 3000]      # all fail
    # beta_MixUCBI_values = [4000, 5000, 6000, 7000, 8000]      # all fail
    beta_MixUCBI_values = [10000, 20000, 40000, 80000, 160000]  # 40000 is the minimum value that seems to work.
    success = np.array([True]*len(beta_MixUCBI_values))
    for beta_MixUCBI in beta_MixUCBI_values:
        print(f"Running MixUCB-I for beta_MixUCBI={beta_MixUCBI}...")
        args = run_mixucbI.parser.parse_args(['--pickle_file', 'simulation_data_spanet.pkl', '--beta_MixUCBI', str(beta_MixUCBI)])
        try:
            run_mixucbI.main(args)
        except cvxpy.error.SolverError as e:
            print(f"SolverError encountered for beta_MixUCBI={beta_MixUCBI}.")
            print(e)
            success[beta_MixUCBI_values.index(beta_MixUCBI)] = False
        print(f"Finished running MixUCB-I for beta_MixUCBI={beta_MixUCBI}.")
    print(beta_MixUCBI_values)
    print(success)
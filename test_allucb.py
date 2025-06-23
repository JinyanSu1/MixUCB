## Simple unit tests for run_allucb.py.
## Manually verify that we create file structure of the form: data_name/seed_{seed}/{mode}_results/{delta}

import numpy as np
import subprocess
from scipy.linalg import inv

# One test for sq_oracle.
def test_sq_oracle():
    print("TEST: sq_oracle")
    T = 5
    seed = 0
    data_name = "synthetic"
    PICKLE_FILE = f"raw_data/multilabel_data_{data_name}_{seed:02d}.pkl"
    cmd = f"python run_allucb.py --T {T} --mode sq_oracle --pickle_file {PICKLE_FILE} --data_name {data_name} --seed {seed}"
    subprocess.run(cmd.split(" "))

# One test for MixUCB.
def test_mixucb():
    print("TEST: mixucb")
    T = 10
    seed = 1
    data_name = "synthetic"
    PICKLE_FILE = f"raw_data/multilabel_data_{data_name}_{seed:02d}.pkl"
    ALPHA = 5
    BETA = 0.1
    DELTA = 4
    cmd = f"python run_allucb.py --T {T} --mode mixI --pickle_file {PICKLE_FILE} --beta {BETA} --alpha {ALPHA} --data_name {data_name} --delta {DELTA} --seed {seed}"
    subprocess.run(cmd.split(" "))

def test_gapscaling():
    """
    Test to see whether the gap scales as expected, as a function of beta-lr and beta-sq.
    Expectation is that it should be directly proportional.
    """
    # simplified version from regression_ucb.
    rng = np.random.default_rng(0)
    # Create random square matrices for X_sum and A. And random context.
    beta_log, beta_sq = 0.1, 0.1
    N=10
    X_sum = rng.random((N, N))
    A = rng.random((N, N))
    context = rng.random(N)
    def compute_sigma(context, X_sum, A, beta_log, beta_sq):
        """
        Compute the sigma value based on the given matrices and beta values.
        """
        combined_cov = X_sum / beta_log**2 + A / beta_sq**2
        sigma = np.sqrt(context.dot(inv(combined_cov).dot(context)))
        return sigma

    original_sigma = compute_sigma(context, X_sum, A, beta_log, beta_sq)

    # Arbitrary scaling factor of 5. Verify that sigma scales in the same way.
    scaling_factor = 5
    beta_log_scaled = beta_log * scaling_factor
    beta_sq_scaled = beta_sq * scaling_factor
    scaled_sigma = compute_sigma(context, X_sum, A, beta_log_scaled, beta_sq_scaled)
    assert np.isclose(original_sigma*scaling_factor, scaled_sigma), f"Expected {original_sigma}*{scaling_factor}, got {scaled_sigma}"

    print("Test passed!")

if __name__=="__main__":
    # test_sq_oracle()
    # test_mixucb()
    test_gapscaling()
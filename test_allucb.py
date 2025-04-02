## Simple unit tests for run_allucb.py.
## Manually verify that we create file structure of the form: data_name/seed_{seed}/{mode}_results/{delta}

import subprocess

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

if __name__=="__main__":
    test_sq_oracle()
    test_mixucb()
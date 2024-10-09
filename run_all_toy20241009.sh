## Runs all the scripts for all 6 baselines.
# fixed params
ALPHA=0.1
LAMBDA=0.001
FILE=simulation_data_toy20241009.pkl
python run_linucb.py --pickle_file $FILE --lambda $LAMBDA --alpha $ALPHA
python run_perfect_expert.py --pickle_file $FILE

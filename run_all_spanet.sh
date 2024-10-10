## Runs all the scripts for all 6 baselines.
# fixed params
TEMPERATURE=$1  # needs to be same temperature as was used for MixUCBI and MixUCBII
ALPHA=$2
# For: spanet dataset
# ALPHA=0.1
# LAMBDA=1
# For: synthetic dataset
# ALPHA=0.1
LAMBDA=0.001    # just use this for now.
T=200
# variable params
# BETA=4000
python run_linucb.py --pickle_file simulation_data_spanet.pkl --lambda $LAMBDA --alpha $ALPHA --T $T
python run_noisy_expert.py --pickle_file simulation_data_spanet.pkl --temperature $TEMPERATURE --T $T
python run_perfect_expert.py --pickle_file simulation_data_spanet.pkl --T $T
# python run_mixucbI.py --pickle_file simulation_data_spanet.pkl   --beta_MixUCBI $BETA --lambda $LAMBDA --alpha $ALPHA --temperature $TEMPERATURE
# python run_mixucbII.py --pickle_file simulation_data_spanet.pkl  --beta_MixUCBII $BETA --lambda $LAMBDA --alpha $ALPHA --temperature $TEMPERATURE
# python run_mixucbIII.py --pickle_file simulation_data_spanet.pkl  --lambda $LAMBDA --alpha $ALPHA
## Runs all the scripts for all 6 baselines.
# fixed params
TEMPERATURE=$1  # needs to be same temperature as was used for MixUCBI and MixUCBII
ALPHA=0.1
LAMBDA=1
# variable params
# BETA=4000
python run_linucb.py --pickle_file simulation_data_spanet.pkl --lambda $LAMBDA --alpha $ALPHA --T 2000
# python run_noisy_expert.py --pickle_file simulation_data_spanet.pkl --temperature $TEMPERATURE
# python run_perfect_expert.py --pickle_file simulation_data_spanet.pkl --T 2000
# python run_mixucbI.py --pickle_file simulation_data_spanet.pkl   --beta_MixUCBI $BETA --lambda $LAMBDA --alpha $ALPHA --temperature $TEMPERATURE
# python run_mixucbII.py --pickle_file simulation_data_spanet.pkl  --beta_MixUCBII $BETA --lambda $LAMBDA --alpha $ALPHA --temperature $TEMPERATURE
# python run_mixucbIII.py --pickle_file simulation_data_spanet.pkl  --lambda $LAMBDA --alpha $ALPHA
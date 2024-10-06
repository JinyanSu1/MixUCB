## Runs all the scripts for all 6 baselines.
# fixed params
TEMPERATURE=0.1
ALPHA=1
# variable params
LAMBDA=10
python run_linucb.py --pickle_file simulation_data_spanet.pkl --lambda $LAMBDA --alpha $ALPHA
python run_noisy_expert.py --pickle_file simulation_data_spanet.pkl --temperature $TEMPERATURE
python run_perfect_expert.py --pickle_file simulation_data_spanet.pkl 
# python run_mixucbI.py --pickle_file simulation_data_spanet.pkl   --beta_MixUCBI 40000
# python run_mixucbII.py --pickle_file simulation_data_spanet.pkl  --beta_MixUCBII 40000
# python run_mixucbIII.py --pickle_file simulation_data_spanet.pkl
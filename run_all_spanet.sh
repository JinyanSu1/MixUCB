## Runs all the scripts for all 6 baselines.
python run_linucb.py --pickle_file simulation_data_spanet.pkl 
python run_noisy_expert.py --pickle_file simulation_data_spanet.pkl 
python run_perfect_expert.py --pickle_file simulation_data_spanet.pkl 
python run_mixucbI.py --pickle_file simulation_data_spanet.pkl   --beta_MixUCBI 40000
python run_mixucbII.py --pickle_file simulation_data_spanet.pkl  --beta_MixUCBII 40000
python run_mixucbIII.py --pickle_file simulation_data_spanet.pkl
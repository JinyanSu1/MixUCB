## for classification datasets
# data_name=MedNIST  # heart_disease
data_name=heart_disease
python generate_multilabel_data.py --data_name ${data_name}
PICKLE_FILE="multilabel_data_${data_name}_42.pkl"

## for rohan's data
# data_name=synthetic
# PICKLE_FILE="raw_data/simulation_data_toy20241009_noise0.2.pkl"

# data_name=spanet
# PICKLE_FILE="raw_data/simulation_data_spanet.pkl"

TEMP=1
BETA=10
ALPHA=10
python run_allucb.py --T 500 --mode lin --pickle_file ${PICKLE_FILE} --alpha ${ALPHA} --data_name ${data_name}
python run_allucb.py --T 500 --mode mixI --pickle_file ${PICKLE_FILE} --beta ${BETA} --alpha ${ALPHA} --data_name ${data_name}
python run_allucb.py --T 500 --mode mixII --pickle_file ${PICKLE_FILE} --beta ${BETA} --alpha ${ALPHA} --data_name ${data_name}
python run_allucb.py --T 500 --mode mixIII --pickle_file ${PICKLE_FILE} --alpha ${ALPHA} --data_name ${data_name}

python run_allucb.py --T 500 --mode sq_oracle --pickle_file ${PICKLE_FILE} --data_name ${data_name}
python run_allucb.py --T 500 --mode lr_oracle --pickle_file ${PICKLE_FILE} --data_name ${data_name}

# python run_noisy_expert.py --pickle_file ${PICKLE_FILE} --temperature ${TEMP} --data_name ${data_name}
# python run_noisy_expert.py --pickle_file ${PICKLE_FILE} --temperature 0.1 --data_name ${data_name}
# python run_perfect_expert.py --pickle_file ${PICKLE_FILE} --data_name ${data_name}
python plot_tools.py --data_name ${data_name}
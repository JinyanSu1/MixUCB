## for classification datasets
# data_name=MedNIST  # heart_disease
# TEMP=1
# BETA=10
# ALPHA=5
# # data_name=heart_disease
# python generate_multilabel_data.py --data_name ${data_name}
# PICKLE_FILE="multilabel_data_${data_name}_42.pkl"

## for rohan's data
TEMP=1
BETA=10
ALPHA=5
# data_name=synthetic
# ORIG_PICKLE_FILE="raw_data/simulation_data_toy20241009_noise0.2.pkl"
# python generate_multilabel_data.py --reprocess ${ORIG_PICKLE_FILE}
# PICKLE_FILE="raw_data/simulation_data_toy20241009_noise0.2reprocessed.pkl"

data_name=spanet
ORIG_PICKLE_FILE="raw_data/simulation_data_spanet.pkl"
python generate_multilabel_data.py --reprocess ${ORIG_PICKLE_FILE}
PICKLE_FILE="raw_data/simulation_data_spanetreprocessed.pkl"


python run_allucb.py --T 500 --mode lin --pickle_file ${PICKLE_FILE} --alpha ${ALPHA} --data_name ${data_name}
python run_allucb.py --T 500 --mode mixI --pickle_file ${PICKLE_FILE} --beta ${BETA} --alpha ${ALPHA} --data_name ${data_name}
python run_allucb.py --T 500 --mode mixII --pickle_file ${PICKLE_FILE} --beta ${BETA} --alpha ${ALPHA} --data_name ${data_name}
python run_allucb.py --T 500 --mode mixIII --pickle_file ${PICKLE_FILE} --alpha ${ALPHA} --data_name ${data_name}

python run_allucb.py --T 500 --mode sq_oracle --pickle_file ${PICKLE_FILE} --data_name ${data_name}
python run_allucb.py --T 500 --mode lr_oracle --pickle_file ${PICKLE_FILE} --data_name ${data_name}

python plot_tools.py --data_name ${data_name}
# data_name=MedNIST  # heart_disease
data_name=heart_disease
python generate_multilabel_data.py --data_name ${data_name}
PICKLE_FILE="multilabel_data_${data_name}_42.pkl"
TEMP=1
BETA=10
ALPHA=5
python run_allucb.py --mode lin --pickle_file ${PICKLE_FILE} --alpha ${ALPHA} --data_name ${data_name}
python run_allucb.py --mode mixI --pickle_file ${PICKLE_FILE} --beta ${BETA} --alpha ${ALPHA} --data_name ${data_name}
python run_allucb.py --mode mixII --pickle_file ${PICKLE_FILE} --beta ${BETA} --alpha ${ALPHA} --data_name ${data_name}
python run_allucb.py --mode mixIII --pickle_file ${PICKLE_FILE} --alpha ${ALPHA} --data_name ${data_name}

# python run_noisy_expert.py --pickle_file ${PICKLE_FILE} --temperature ${TEMP} --data_name ${data_name}
# python run_noisy_expert.py --pickle_file ${PICKLE_FILE} --temperature 0.1 --data_name ${data_name}
# python run_perfect_expert.py --pickle_file ${PICKLE_FILE} --data_name ${data_name}
python plot_tools.py --data_name ${data_name}
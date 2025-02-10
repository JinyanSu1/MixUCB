# data_name=MedNIST  # heart_disease
data_name=heart_disease
python generate_multilabel_data.py --data_name ${data_name}
PICKLE_FILE="multilabel_data_${data_name}_42.pkl"
TEMP=1
BETA=10
ALPHA=1
# python run_linucb.py --pickle_file ${PICKLE_FILE} --alpha ${ALPHA} --data_name ${data_name}
# python run_mixucbI.py --pickle_file ${PICKLE_FILE} --temperature ${TEMP} --beta_MixUCBI ${BETA} --alpha ${ALPHA} --data_name ${data_name}
python run_mixucbII.py --pickle_file ${PICKLE_FILE} --temperature ${TEMP} --beta_MixUCBII ${BETA} --alpha ${ALPHA} --data_name ${data_name}
# python run_mixucbIII.py --pickle_file ${PICKLE_FILE} --alpha ${ALPHA} --data_name ${data_name}
# python run_noisy_expert.py --pickle_file ${PICKLE_FILE} --temperature ${TEMP} --data_name ${data_name}
# python run_noisy_expert.py --pickle_file ${PICKLE_FILE} --temperature 0.1 --data_name ${data_name}
# python run_perfect_expert.py --pickle_file ${PICKLE_FILE} --data_name ${data_name}
python plot_tools.py --data_name ${data_name}
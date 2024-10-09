PICKLE_FILE='multilabel_data_MedNIST_42.pkl'

python run_linucb.py --pickle_file ${PICKLE_FILE}
python run_mixucbI.py --pickle_file ${PICKLE_FILE}
python run_mixucbII.py --pickle_file ${PICKLE_FILE}
python run_mixucbIII.py --pickle_file ${PICKLE_FILE}
python run_noisy_expert.py --pickle_file ${PICKLE_FILE}
python run_perfect_expert.py --pickle_file ${PICKLE_FILE}
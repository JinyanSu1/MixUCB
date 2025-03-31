# (1) Data generation/hyperparameter settings

SEEDS='42 0 1 2 3'

## for classification datasets
DELTA='4 5 6'
# data_name=MedNIST  # heart_disease
data_name=heart_disease
for seed in ${SEEDS}; do
    python generate_multilabel_data.py --data_name ${data_name} --seed ${seed}
done
# PICKLE_FILE="multilabel_data_${data_name}_42.pkl"

## for rohan's synthetic data
DELTA='2 3 4'
data_name=synthetic
noise_std='0.2'
for seed in ${SEEDS}; do
    python generate_multilabel_data.py --data_name ${data_name} --seed ${seed} --noise_std ${noise_std}
done

## for spanet data.
DELTA='3 4 5'
data_name=spanet
pca_dim='5'
for seed in ${SEEDS}; do
    python generate_multilabel_data.py --data_name ${data_name} --seed ${seed} --n_features ${pca_dim}
done

TEMP=1
BETA=10
ALPHA=5

# (2) Algorithm execution

# python run_allucb.py --T 300 --mode lin --pickle_file ${PICKLE_FILE} --alpha ${ALPHA} --data_name ${data_name}
# python run_allucb.py --T 300 --mode mixI --pickle_file ${PICKLE_FILE} --beta ${BETA} --alpha ${ALPHA} --data_name ${data_name} --delta ${DELTA}
# python run_allucb.py --T 300 --mode mixII --pickle_file ${PICKLE_FILE} --beta ${BETA} --alpha ${ALPHA} --data_name ${data_name} --delta ${DELTA}
# python run_allucb.py --T 300 --mode mixIII --pickle_file ${PICKLE_FILE} --alpha ${ALPHA} --data_name ${data_name} --delta ${DELTA}

# python run_allucb.py --T 300 --mode sq_oracle --pickle_file ${PICKLE_FILE} --data_name ${data_name}
# python run_allucb.py --T 300 --mode lr_oracle --pickle_file ${PICKLE_FILE} --data_name ${data_name}

# (3) Plotting

# ROHAN: assumes that the {data_name} directory exists.
# python plot_tools.py --data_name ${data_name}
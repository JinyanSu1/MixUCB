#!/bin/bash
## GENERAL SCRIPT STRUCTURE
# 0. Random seeds
# 1. Data generation/hyperparameter settings
# 2. Algorithm execution
# 3. Plotting
####################################

# (0) Random seeds

SEEDS='42 0 1 2 3'
# SEEDS='42'
# SEEDS='3'

# (1) Data generation/hyperparameter settings

# ## for classification datasets
# DELTA='4 5 6'
# data_names='MedNIST heart_disease'
# for data_name in ${data_names}; do
#     for seed in ${SEEDS}; do
#         python generate_multilabel_data.py --data_name ${data_name} --seed ${seed}
#     done
# done

# ## for rohan's synthetic data
# DELTA='2 3 4'
# data_name=synthetic
# noise_std='0.2'
# for seed in ${SEEDS}; do
#     python generate_multilabel_data.py --data_name ${data_name} --seed ${seed} --noise_std ${noise_std}
# done

# ## for spanet data.
# DELTA='3 4 5'
# data_name=spanet
# pca_dim='5'
# for seed in ${SEEDS}; do
#     python generate_multilabel_data.py --data_name ${data_name} --seed ${seed} --n_features ${pca_dim}
# done

# TEMP=1              # NOTE: not used.
# ALPHA=5
# BETA=10

# (2) Algorithm execution
# Loop through all datasets and all seeds.

data_names_all='MedNIST heart_disease synthetic spanet'
# data_names_all='synthetic'
for data_name in ${data_names_all}; do
    if [ ${data_name} == 'MedNIST' ]; then
        # DELTA='4 5 6'
        # ALPHA='5'             # alpha_orig
        # BETA='10'              # beta_orig
        DELTA='0.5 0.625 0.75'
        ALPHA='0.625'
        BETA='1.25'
        LAMBDA='0.001'          # ORIGINAL LAMBDA.    
    elif [ ${data_name} == 'heart_disease' ]; then
        # DELTA='4 5 6'
        # ALPHA='5'             # alpha_orig
        # BETA='10'              # beta_orig
        DELTA='0.5 0.625 0.75'
        ALPHA='0.625'
        BETA='1.25'
        LAMBDA='0.001'          # ORIGINAL LAMBDA.    
    elif [ ${data_name} == 'synthetic' ]; then
        # DELTA='2 3 4'
        # ALPHA='5'             # alpha_orig
        # BETA='10'              # beta_orig
        DELTA='0.5 0.75 1.0'    # Let C=4. delta_orig/C
        ALPHA='1.25'             # alpha_orig/C
        BETA='2.5'              # beta_orig/C
        LAMBDA='0.001'          # ORIGINAL LAMBDA.    
    elif [ ${data_name} == 'spanet' ]; then
        DELTA='3 4 5'
        ALPHA='5'             # alpha_orig
        BETA='10'              # beta_orig
        DELTA='0.5 0.625 0.75'
        ALPHA='0.625'
        BETA='1.25'
        LAMBDA='0.001'          # ORIGINAL LAMBDA.    
    fi
    for seed in ${SEEDS}; do
        formatted_seed=$(printf "%02d" $seed)
        PICKLE_FILE="raw_data/multilabel_data_${data_name}_${formatted_seed}.pkl"        
        python run_allucb.py --T 300 --mode lin --pickle_file ${PICKLE_FILE} --alpha ${ALPHA} --data_name ${data_name} --seed ${seed} --lambda_ ${LAMBDA}
        python run_allucb.py --T 300 --mode mixI --pickle_file ${PICKLE_FILE} --beta ${BETA} --alpha ${ALPHA} --data_name ${data_name} --delta ${DELTA} --seed ${seed} --lambda_ ${LAMBDA}
        python run_allucb.py --T 300 --mode mixII --pickle_file ${PICKLE_FILE} --beta ${BETA} --alpha ${ALPHA} --data_name ${data_name} --delta ${DELTA} --seed ${seed} --lambda_ ${LAMBDA}
        python run_allucb.py --T 300 --mode mixIII --pickle_file ${PICKLE_FILE} --alpha ${ALPHA} --data_name ${data_name} --delta ${DELTA} --seed ${seed} --lambda_ ${LAMBDA}
        python run_allucb.py --T 300 --mode sq_oracle --pickle_file ${PICKLE_FILE} --data_name ${data_name} --seed ${seed} --lambda_ ${LAMBDA}
        python run_allucb.py --T 300 --mode lr_oracle --pickle_file ${PICKLE_FILE} --data_name ${data_name} --seed ${seed} --lambda_ ${LAMBDA}
        python run_allucb.py --T 300 --mode perfect_exp --pickle_file ${PICKLE_FILE} --data_name ${data_name} --seed ${seed} --lambda_ ${LAMBDA}
        python run_allucb.py --T 300 --mode noisy_exp --pickle_file ${PICKLE_FILE} --data_name ${data_name} --seed ${seed} --lambda_ ${LAMBDA}
    done
done

# (3) Plotting

# ROHAN: assumes that the {data_name} directory exists.
for data_name in ${data_names_all}; do
    python plot_tools.py --data_name ${data_name} --seeds ${SEEDS}
done
# data_name=synthetic
# python plot_tools.py --data_name ${data_name} --seeds ${SEEDS}
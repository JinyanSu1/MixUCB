## Uses run_all_spanet for baselines.

# SAVENAME_1="g2temp1.0_explinearoracle_20241009"
# SAVENAME_1="g2temp1.0_fixseed_20241009"
# SAVENAME_5="g2temp5.0_fixseed_20241009"

# SAVENAME_1="synthetictemp1.0_20241009_5"
# SAVENAME_5="synthetictemp5.0_20241009_5"

# SAVENAME_5="spanettemp5.0_20241010"
# SAVENAME_5="spanettemp5.0_20241010_2"
# SAVENAME_5="spanettemp5.0_20241010_3" # T=200, beta from 50...
# SAVENAME_5="spanettemp5.0_20241010_4" # T=150
# SAVENAME_5="spanettemp5.0_20241010_5"   # T=100
SAVENAME_5_ALPHA01="spanettemp5.0_alpha0.1_20241010_1"
SAVENAME_5_ALPHA005="spanettemp5.0_alpha0.05_20241010_1"

# python tune_mixUCB_analyze.py --temperature 1.0 > output1.0.txt
# python tune_mixUCB_analyze.py --temperature 5.0 --alpha 0.1 > output5.0.txt
python tune_mixUCB_analyze.py --temperature 5.0 --alpha 0.1 > output0.1.txt
python tune_mixUCB_analyze.py --temperature 5.0 --alpha 0.05 > output0.05.txt
# mkdir $SAVENAME_1
# mkdir $SAVENAME_5
mkdir $SAVENAME_5_ALPHA01
mkdir $SAVENAME_5_ALPHA005
# mv *_results_temp1.0* $SAVENAME_1
# mv *_results_temp5.0* $SAVENAME_5
mv *_results_temp5.0_alpha0.1* $SAVENAME_5_ALPHA01
mv *_results_temp5.0_alpha0.05* $SAVENAME_5_ALPHA005
# mv output1.0.txt $SAVENAME_1
# mv output5.0.txt $SAVENAME_5
mv output0.1.txt $SAVENAME_5_ALPHA01
mv output0.05.txt $SAVENAME_5_ALPHA005
# mv *1.0.pkl $SAVENAME_1
# mv *5.0_0.1.pkl $SAVENAME_5
mv *5.0_0.1.pkl $SAVENAME_5_ALPHA01
mv *5.0_0.05.pkl $SAVENAME_5_ALPHA005
# mkdir -p Figures/$SAVENAME_1
# mkdir -p Figures/$SAVENAME_5
mkdir -p Figures/$SAVENAME_5_ALPHA01
mkdir -p Figures/$SAVENAME_5_ALPHA005
# bash run_all_spanet.sh 1.0
# mv linucb_results_0/ noisy_expert_results/ perfect_expert_results_0/ $SAVENAME_1
# bash run_all_spanet.sh 5.0 0.1
# mv linucb_results_0/ noisy_expert_results/ perfect_expert_results_0/ $SAVENAME_5

# Need to be careful about what lambda value you use.
# bash run_all_spanet.sh 5.0 0.1
# mv linucb_results_0/ noisy_expert_results/ perfect_expert_results_0/ $SAVENAME_5_ALPHA01
# bash run_all_spanet.sh 5.0 0.05
# mv linucb_results_0/ noisy_expert_results/ perfect_expert_results_0/ $SAVENAME_5_ALPHA005
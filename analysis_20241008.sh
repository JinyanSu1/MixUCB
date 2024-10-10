# SAVENAME_1="g2temp1.0_explinearoracle_20241009"
# SAVENAME_1="g2temp1.0_fixseed_20241009"
# SAVENAME_5="g2temp5.0_fixseed_20241009"

SAVENAME_1="synthetictemp1.0_20241009_5"
SAVENAME_5="synthetictemp5.0_20241009_5"

python tune_mixUCB_analyze.py --temperature 1.0 > output1.0.txt
python tune_mixUCB_analyze.py --temperature 5.0 > output5.0.txt
mkdir $SAVENAME_1
mkdir $SAVENAME_5
mv *_results_temp1.0* $SAVENAME_1
mv *_results_temp5.0* $SAVENAME_5
mv output1.0.txt $SAVENAME_1
mv output5.0.txt $SAVENAME_5
mv *1.0.pkl $SAVENAME_1
mv *5.0.pkl $SAVENAME_5
mkdir -p Figures/$SAVENAME_1
mkdir -p Figures/$SAVENAME_5
bash run_all_spanet.sh 1.0
mv linucb_results_0/ noisy_expert_results/ perfect_expert_results_0/ $SAVENAME_1
bash run_all_spanet.sh 5.0
mv linucb_results_0/ noisy_expert_results/ perfect_expert_results_0/ $SAVENAME_5
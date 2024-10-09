SAVENAME_1="g2temp1.0_explinearoracle_20241009"
SAVENAME_5="g2temp5.0_explinearoracle_20241009"

# python tune_mixUCB_analyze.py --temperature 1.0 > output1.0.txt
# python tune_mixUCB_analyze.py --temperature 5.0 > output5.0.txt
# mkdir $SAVENAME_1
# mkdir $SAVENAME_5
# mv *_results_temp1.0* $SAVENAME_1
# mv *_results_temp5.0* $SAVENAME_5
# mkdir -p Figures/$SAVENAME_1
# mkdir -p Figures/$SAVENAME_5
# bash run_all_spanet.sh 1.0
# mv linucb_results_0/ noisy_expert_results/ perfect_expert_results/ $SAVENAME_1
# bash run_all_spanet.sh 5.0
# mv linucb_results_0/ noisy_expert_results/ perfect_expert_results/ $SAVENAME_5
SAVENAME_001="syntheticalpha0.01_20241010"
SAVENAME_1="syntheticalpha1_20241010"

# python tune_mixUCB_analyze.py --temperature 5.0 --alpha 0.01 > output0.01.txt
# python tune_mixUCB_analyze.py --temperature 5.0 --alpha 1.0 > output1.txt
# mkdir $SAVENAME_001
# mkdir $SAVENAME_1
# mv *_results_temp5.0_alpha0.01* $SAVENAME_001
# mv *_results_temp5.0_alpha1.0* $SAVENAME_1
# mv output0.01.txt $SAVENAME_001
# mv output1.txt $SAVENAME_1
# mv *0.01.pkl $SAVENAME_001
# mv *1.0.pkl $SAVENAME_1
# mkdir -p Figures/$SAVENAME_001
# mkdir -p Figures/$SAVENAME_1
bash run_all_synthetic.sh 5.0 0.01
mv linucb_results_0/ noisy_expert_results/ perfect_expert_results_0/ $SAVENAME_001
bash run_all_synthetic.sh 5.0 1.0
mv linucb_results_0/ noisy_expert_results/ perfect_expert_results_0/ $SAVENAME_1
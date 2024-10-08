# python tune_mixUCB_analyze.py --temperature 1.0 > output1.0.txt
# python tune_mixUCB_analyze.py --temperature 5.0 > output5.0.txt
# mkdir g2temp1.0_linearreward_20241008
# mkdir g2temp5.0_linearreward_20241008
# mv *_results_temp1.0* g2temp1.0_linearreward_20241008
# mv *_results_temp5.0* g2temp5.0_linearreward_20241008
# mkdir -p Figures/g2temp1.0_linearreward_20241008
# mkdir -p Figures/g2temp5.0_linearreward_20241008
# bash run_all_spanet.sh 1.0
# mv linucb_results_0/ noisy_expert_results/ perfect_expert_results/ g2temp1.0_linearreward_20241008/
# bash run_all_spanet.sh 5.0
# mv linucb_results_0/ noisy_expert_results/ perfect_expert_results/ g2temp5.0_linearreward_20241008/
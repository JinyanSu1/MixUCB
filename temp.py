
# Want to repeatedly execute the following:
# i from 0 to 4.
# mv mixucbI_results_temp5.0_alpha0.1_{i}/0.75/ spanettemp5.0_202410
# 10_2/mixucbI_results_temp5.0_alpha0.1_{i}/

import os

for i in range(5):
    os.system(f"mv mixucbI_results_temp5.0_alpha0.1_{i}/0.75/ spanettemp5.0_20241010_2/mixucbI_results_temp5.0_alpha0.1_{i}/")
    os.system(f"mv mixucbII_results_temp5.0_alpha0.1_{i}/0.75/ spanettemp5.0_20241010_2/mixucbII_results_temp5.0_alpha0.1_{i}/")
    os.system(f"mv mixucbIII_results_temp5.0_alpha0.1_{i}/0.75/ spanettemp5.0_20241010_2/mixucbIII_results_temp5.0_alpha0.1_{i}/")

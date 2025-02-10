# All the baselines


Generate the and store the data for T rounds using ```generate_multilabel_data.py``` 

## Installation
```bash
conda create -f env.yml
```
## mosek

```bash
# https://www.mosek.com/resources/getting-started/
cp -r mosek ~/mosek
```

# Baselines
```bash
python run_linucb.py
python run_noisy_expert.py
python run_perfect_expert.py
python run_mixucbI.py
python run_mixucbII.py
python run_mixucbIII.py
```
Or
```bash
bash run_all.sh
```

# D4RL
```bash
pip install "cython<3"
pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl
```


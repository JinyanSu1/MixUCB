# All the baselines


Generate the and store the data for T rounds using ```generate_all_data.py``` 

Install:
```
conda create -n mixucb python=3.10
conda activate mixucb
pip install -r requirements.txt 
# Install pytorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# Install multiprocess
pip install multiprocess
```

Set up MOSEK.

# Baselines
```bash
python run_linucb.py
python run_noisy_expert.py  # requires torch
python run_perfect_expert.py
python run_mixucbI.py       # requires torch
python run_mixucbII.py      # requires torch, multiprocess
python run_mixucbIII.py

```

# D4RL
```bash
pip install "cython<3"
pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl
```


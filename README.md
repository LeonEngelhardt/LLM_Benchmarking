# LLM Benchmark Framework

This repository acts as a framework to benchmark LLMs with the following three experiments:

1. One-Shot In-Context Learning  
2. Two-Shot In-Context Learning  
3. Learning from Experience  

The evaluation includes the following criterias:

- strict correctness (answer given by LLM = ground truth)  
- Closeness-Score (1-10) via ranking of another LLM of the given answer compared to the ground truth  

## Installation

```bash
pip install -r requirements.txt


## Running the experiments

## Run all experiments:
python -m src.main

## Only One-Shot:
python -m src.main --experiment one-shot

## Only Two-Shot:
python -m src.main --experiment two-shot

## Only Learning from Experience:
python -m src.main --experiment lfe

## It is also possible to only run a single, specified model
# just add the argument: --model gpt2    --> if you would like to test only gpt2
python -m src.main --model gpt2



## running the framework on a cluster:
# create two virtual environments. The first one:
python -m venv venv_all_other_models

# activate it:
venv_all_other_models\Scripts\activate

# install all needed modules for it:
pip istall requirements.txt

# deactivate the first venv to create the second one:
deactivate

# create the second one:
python -m venv venv_only_deepseek_vl2

# activate it
venv_only_deepseek_vl2\Scripts\activate

# change path to repo deepseek folder
cd external
cd deepseek-vl2

# install all needed modules for it: 
pip install -e .
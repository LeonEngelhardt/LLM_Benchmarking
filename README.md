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
# running the framework on a cluster:

# !! Log in into the cluster FIRST !! see [https://wiki.bwhpc.de/e/BwUniCluster2.0](https://wiki.bwhpc.de/e/BwUniCluster3.0)

# create two virtual environments. The first one:
python -m venv venv_all_other_models

# activate it:
venv_all_other_models\Scripts\activate

# install all needed modules for it:
pip istall requirements.txt

# deactivate the first venv to create the second one:
deactivate

# ONLY FOR  deepseek_v2 !!!create the second one (second environement) :
python -m venv venv_only_deepseek_vl2

# ONLY FOR  deepseek_v2 !!! activate it
venv_only_deepseek_vl2\Scripts\activate

# ONLY FOR  deepseek_v2 !!! change path to repo deepseek folder
cd external
cd deepseek-vl2

# install all needed modules for it: 
pip install -e .

## Running the experiments

## Run all experiments (all models):
python -m src.main

## Only One-Shot (one model):
python -m src.main --model gpt2 -experiment one-shot

## Only Two-Shot(one model):
python -m src.main --model gpt2 --experiment two-shot

## Only Learning from Experience(one model):
python -m src.main -model gpt2 --experiment lfe

##One-Shot:  + Two-Shot: Learning from Experience:
python -m src.main --model gpt2

## It is also possible to only run a single, specified model
# just add the argument: --model gpt2    --> if you would like to test only gpt2
python -m src.main --model gpt2


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

Run all experiments:
python -m src.main

Only One-Shot:
python -m src.main --experiment one-shot

Only Two-Shot:
python -m src.main --experiment two-shot

Only Learning from Experience:
python -m src.main --experiment lfe
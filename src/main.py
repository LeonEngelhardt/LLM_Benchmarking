import os
import torch
from dotenv import load_dotenv
from src.utils import load_csv, save_csv
from src.llm_models import HuggingFaceLLM
from src.benchmark import BenchmarkRunner
from src.evaluator import ClosenessEvaluator, strict_match

load_dotenv()
hf_token = os.environ.get("HF_TOKEN")

df_all = load_csv("data/dataset.csv")
closeness_eval = ClosenessEvaluator()

models_to_test = [
    {"name": "gpt2", "vision": False},
    {"name": "Salesforce/blip-image-captioning-base", "vision": True},
    #{"name": "meta-llama/Llama-2-7b-chat-hf", "vision": False},
    #{"name": "deepseek-ai/deepseek-vl-7b-chat", "vision": True},
]

for model_info in models_to_test:
    model_name = model_info["name"]
    vision_enabled = model_info["vision"]

    print(f"\n=== Benchmarking {model_name} (Vision: {vision_enabled}) ===")

    if vision_enabled:
        df = df_all[df_all["image_path"].notna() & (df_all["image_path"].str.strip() != "")].reset_index(drop=True)
        print(f"[INFO] Vision-Model → {len(df)} Picture-Questions")
    else:
        df = df_all[df_all["image_path"].isna() | (df_all["image_path"].str.strip() == "")].reset_index(drop=True)
        print(f"[INFO] Text-Model → {len(df)} Text-Questions")

    if len(df) == 0:
        print("[WARNING] Skipping this model, because no suited questions are loaded.")
        continue

    llm = HuggingFaceLLM(
        model_name=model_name,
        hf_token=hf_token,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    llm.load_model()

    runner = BenchmarkRunner(
        df=df,
        llm=llm,
        evaluator=strict_match,
        closeness_evaluator=closeness_eval,
        vision=vision_enabled
    )

    # One-Shot
    print(f"--- {model_name} One-Shot ---")
    one_shot_df = runner.run_one_shot()
    save_csv(one_shot_df, f"results/{model_name.replace('/', '_')}_one_shot.csv")

    # Two-Shot
    print(f"--- {model_name} Two-Shot ---")
    two_shot_df = runner.run_two_shot()
    save_csv(two_shot_df, f"results/{model_name.replace('/', '_')}_two_shot.csv")

    # Learning-from-Experience
    print(f"--- {model_name} Learning-from-Experience ---")
    lfe_df = runner.run_learning_from_experience()
    save_csv(lfe_df, f"results/{model_name.replace('/', '_')}_lfe.csv")
import os
from dotenv import load_dotenv
from src.utils import load_csv, save_csv
from src.benchmark import BenchmarkRunner
from src.evaluator import (
    strict_match,
    ClosenessEvaluator,
    LLMClosenessEvaluator
)
from src.llm_backends.factory import get_llm

# Setup
load_dotenv()
df_all = load_csv("data/dataset.csv")

# Closeness Evaluator
# Use Qwen as judge IF API key exists, otherwise fallback (e.g. for local testing)
if os.getenv("OPENROUTER_API_KEY"):
    print("[INFO] Using Qwen as LLM-based closeness evaluator")

    qwen_judge = get_llm(
        model_name="qwen/qwen3-235b",
        vision=False,
        #verbose=True
    )
    closeness_eval = LLMClosenessEvaluator(qwen_judge)
else:
    print("[INFO] Using string-based closeness evaluator (local fallback)")
    closeness_eval = ClosenessEvaluator()


# Models to benchmark
models_to_test = [
    # Text-only
    {"name": "gpt2", "vision": False},                                  # local HF
    # {"name": "meta-llama/Llama-2-7b-chat-hf", "vision": False},       # local HF
    # {"name": "mistral/mixtral-8x7b", "vision": False},                # OpenRouter
    # {"name": "qwen/qwen3-235b", "vision": False},                     # OpenRouter
    # {"name": "gpt-4o", "vision": False},                              # OpenAI --> can support vision if we use the right endpoint, otherwise only text
    # {"name": "claude-3-opus", "vision": False},                       # Anthropic
    # {"name": "google/gemma-2-9b-it", "vision": False}                 # HF
    # {"name": "google/gemma-2-27b-it", "vision": False}                # HF
    # {"name": "internlm/internlm2-7b-chat", "vision": False}           # HF
    # {"name": "internlm/Intern-S1", "vision": False}                   # HF

    # Vision
    {"name": "Salesforce/blip-image-captioning-base", "vision": True},  # local HF
    # {"name": "qwen/qwen-vl", "vision": True},                         # OpenRouter
    # {"name": "deepseek-r1", "vision": True},                          # OpenRouter
    # {"name": "internvl/intern-s1", "vision": True},                   # HF / API
    # {"name": "gemini-2.5-pro", "vision": True}
    # {"name": "gemini-2.5-flash", "vision": True},
    # {"name": "llava-hf/llava-1.6-mistral-7b-hf", "vision": True}
    # {"name": "BAAI/Aquila-VL-2B-llava-qwen", "vision": True}
]

# Benchmarking loop
for model_info in models_to_test:
    model_name = model_info["name"]
    vision_enabled = model_info["vision"]

    print(f"\n=== Benchmarking {model_name} (Vision={vision_enabled}) ===")

    # Filter dataset
    """
    if vision_enabled:
        df = df_all[
            df_all["image_path"].notna()
            & (df_all["image_path"].str.strip() != "")
        ].reset_index(drop=True)
        print(f"[INFO] Vision model → {len(df)} image questions")
    else:
        df = df_all[
            df_all["image_path"].isna()
            | (df_all["image_path"].str.strip() == "")
        ].reset_index(drop=True)
        print(f"[INFO] Text model → {len(df)} text questions")

    if df.empty:
        print("[WARNING] No suitable questions found → skipping model")
        continue"""
    
    # Filter dataset --> second possibility
    if vision_enabled:
        # Vision models receive all questions
        df = df_all.reset_index(drop=True)
        print(f"[INFO] Vision model → {len(df)} total questions (text + image)")
    else:
        # Text models receive only text questions (questions with images are filtered out)
        df = df_all[
            df_all["image_path"].isna()
            | (df_all["image_path"].str.strip() == "")
        ].reset_index(drop=True)
        print(f"[INFO] Text model → {len(df)} text questions")

    if df.empty:
        print("[WARNING] No suitable questions found → skipping model")
        continue



    # Load model via factory
    llm = get_llm(
        model_name=model_name,
        vision=vision_enabled,
        #verbose=True
    )
    llm.load()

    runner = BenchmarkRunner(
        df=df,
        llm=llm,
        evaluator=strict_match,
        closeness_evaluator=closeness_eval,
        vision=vision_enabled
    )

    # One-Shot
    print(f"--- {model_name} | One-Shot ---")
    one_shot_df = runner.run_one_shot()
    save_csv(
        one_shot_df,
        f"results/{model_name.replace('/', '_')}_one_shot.csv"
    )

    # Two-Shot
    print(f"--- {model_name} | Two-Shot ---")
    two_shot_df = runner.run_two_shot()
    save_csv(
        two_shot_df,
        f"results/{model_name.replace('/', '_')}_two_shot.csv"
    )

    # Learning-from-Experience
    print(f"--- {model_name} | Learning-from-Experience ---")
    lfe_df = runner.run_learning_from_experience(max_iterations=5)
    save_csv(
        lfe_df,
        f"results/{model_name.replace('/', '_')}_lfe.csv"
    )
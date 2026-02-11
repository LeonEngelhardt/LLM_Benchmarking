import os
import argparse
from dotenv import load_dotenv
from src.utils import load_csv, save_csv
from src.benchmark import BenchmarkRunner
from src.evaluator import (
    strict_match,
    ClosenessEvaluator,
    LLMClosenessEvaluator
)
from src.llm_backends.factory import get_llm


def main():

    parser = argparse.ArgumentParser(description="LLM Benchmark Framework")

    parser.add_argument(
        "--experiment",
        type=str,
        choices=["one-shot", "two-shot", "lfe", "all"],
        default="all",
        help="Which experiment to run"
    )

    args = parser.parse_args()


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
        {"name": "gpt2", "vision": False},                                  # local HF --> only for testing
                    # {"name": "qwen/qwen3-235b", "vision": False},                     # OpenRouter API
        # {"name": "mistralai/Mistral-7B-Instruct-v0.3", "vision": False},  # HF
        # {"name": "claude-3-opus", "vision": False},                       # Anthropic API
        # {"name": "google/gemma-2-9b-it", "vision": False}                 # HF
        # {"name": "google/gemma-2-27b-it", "vision": False}                # HF (needs strong gpu i.e. at least 48 GB of GPU advised --> keep it?)

        # Vision
        {"name": "Salesforce/blip-image-captioning-base", "vision": True},  # local HF --> only for testing
        # {"name": "deepseek-r1", "vision": True},                          # OpenRouter API
        # {"name": "gemini-2.5-pro", "vision": True},                       # Gemini API
        # {"name": "gemini-2.5-flash", "vision": True},                     # Gemini API --> but faster and cheaper than the pro version
        # {"name": "llava-hf/llava-v1.6-mistral-7b-hf", "vision": True}     # HF
        # {"name": "internlm/Intern-S1", "vision": True}                    # HF
        # {"name": "gpt-5.1-chat-latest", "vision": True},                  # OpenAI
        # {"name": "meta-llama/Llama-4-Scout-17B-16E-Instruct", "vision": True},       # HF local / HF inference
        # {"name": "Qwen/Qwen3-VL-235B-A22B-Instruct", "vision": True},     # HF
    ]

    # Todo: gemini, deepseek, claude and gemma (gwen3-235b)

    # Benchmarking loop
    for model_info in models_to_test:
        model_name = model_info["name"]
        vision_enabled = model_info["vision"]

        print(f"\n=== Benchmarking {model_name} (Vision={vision_enabled}) ===")
        
        # Filter dataset
        if vision_enabled:
            # Vision models receive all questions
            df = df_all.reset_index(drop=True)
            print(f"[INFO] Vision model -> {len(df)} total questions (text + image)")
        else:
            # Text models receive only text questions (questions with images are filtered out)
            df = df_all[
                df_all["image_path"].isna()
                | (df_all["image_path"].str.strip() == "")
            ].reset_index(drop=True)
            print(f"[INFO] Text model -> {len(df)} text questions")

        if df.empty:
            print("[WARNING] No suitable questions found -> skipping model")
            continue



        # Load model via factory
        llm = get_llm(
            model_name=model_name,
            vision=vision_enabled,
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
        if args.experiment in ["one-shot", "all"]:
            print(f"--- {model_name} | One-Shot ---")
            one_shot_df = runner.run_one_shot()
            save_csv(
                one_shot_df,
                f"results/{model_name.replace('/', '_')}_one_shot.csv"
            )

        # Two-Shot
        if args.experiment in ["two-shot", "all"]:
            print(f"--- {model_name} | Two-Shot ---")
            two_shot_df = runner.run_two_shot()
            save_csv(
                two_shot_df,
                f"results/{model_name.replace('/', '_')}_two_shot.csv"
            )

        # Learning-from-Experience
        if args.experiment in ["lfe", "all"]:
            print(f"--- {model_name} | Learning-from-Experience ---")
            lfe_df = runner.run_learning_from_experience(max_iterations=5)
            save_csv(
                lfe_df,
                f"results/{model_name.replace('/', '_')}_lfe.csv"
            )


if __name__ == "__main__":
    main()
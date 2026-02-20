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


def get_active_venv():
    venv_name = os.path.basename(os.getenv('VIRTUAL_ENV', ''))
    return venv_name


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
            model_name="Qwen/Qwen3-VL-235B-A22B-Instruct",
            vision=False,
        )
        qwen_judge.load()

        closeness_eval = LLMClosenessEvaluator(qwen_judge)
    else:
        print("[INFO] Using string-based closeness evaluator (local fallback)")
        closeness_eval = ClosenessEvaluator()


    prompt_rewriter_llm = None

    if os.getenv("OPENROUTER_API_KEY"):
        print("[INFO] Loading Qwen3 as Prompt Rewriter")

        prompt_rewriter_llm = get_llm(
            model_name="Qwen/Qwen3-VL-235B-A22B-Instruct",
            vision=False,
        )
        prompt_rewriter_llm.load()
    else:
        print("[INFO] No API key found -> Prompt rewriting disabled")


    venv_name = get_active_venv()

    # Models to benchmark
    if venv_name == "venv_deepseek_vl2":
            pass
            # models_to_test = [ {"name": "deepseek-ai/deepseek-vl2", "vision": True} ]
    elif venv_name == "venv_all_other_models":
        models_to_test = [
            # Text-only
             {"name": "gpt2", "vision": False},                                    # local HF --> only for testing
            # {"name": "mistralai/Mistral-7B-Instruct-v0.3", "vision": False},      # HF   
            # {"name": "deepseek-v3.2", "vision": False},                           # Deepseek API
            # {"name": "DeepSeek-V3.1", "vision": False},
            # {"name": "DeepSeek-V3", "vision": False},
            # {"name": "DeepSeek-V2", "vision": False},  
            # {"name": "Salesforce/blip-image-captioning-base", "vision": True},    # local HF --> only for testing        
            # {"name": "llava-hf/llava-onevision-qwen2-7b-ov-hf", "vision": True}   # HF
            # {"name": "internlm/Intern-S1", "vision": True},                       # HF
            # {"name": "claude-opus-4-6", "vision": True},                          # Anthropic API
            # {"name": "claude-3-opus-latest", "vision": True},                     # Anthropic API
            # {"name": "gpt-5.2", "vision": True},                                  # OpenAI
            # {"name": "gpt-4.1", "vision": True},                                  # OpenAI 
            # {"name": "gpt-3.5-turbo", "vision": False},                           # OpenAI 
            # {"name": "Qwen/Qwen3-VL-235B-A22B-Instruct", "vision": True},         # HF
            # {"name": "Qwen/Qwen2.5-VL-32B-Instruct", "vision": True},             # HF
            # {"name": "Qwen/Qwen2-VL-2B-Instruct", "vision": True},                # HF
            # {"name": "meta-llama/Llama-4-Scout-17B-16E-Instruct", "vision": True},# HF local / HF inference
            # {"name": "meta-llama/Llama-3.2-90B-Vision-Instruct", "vision": True},  
            # {"name": "meta-llama/Llama-3.1-70B-Instruct", "vision": False},  
            # {"name": "meta-llama/Meta-Llama-3-70B-Instruct", "vision": False},  
            # {"name": "google/gemma-3-27b-it", "vision": True},                    # HF
            # {"name": "google/gemma-2-9b-it", "vision": False},                    # HF
            # {"name": "gemini-3-pro-preview", "vision": True},                     # Gemini API
            # {"name": "gemini-2.5-pro", "vision": True},                           # Gemini API
        ]

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


        #active_prompt_rewriter = None
        #if not vision_enabled:
        #    active_prompt_rewriter = prompt_rewriter_llm

        runner = BenchmarkRunner(
            df=df,
            llm=llm,
            evaluator=strict_match,
            closeness_evaluator=closeness_eval,
            vision=vision_enabled,
            prompt_rewriter_llm=prompt_rewriter_llm #active_prompt_rewriter
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
            lfe_df = runner.run_learning_from_experience()
            save_csv(
                lfe_df,
                f"results/{model_name.replace('/', '_')}_lfe.csv"
            )


if __name__ == "__main__":
    main()
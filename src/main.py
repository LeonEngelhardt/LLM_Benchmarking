import os
import torch
import argparse
import glob
from contextlib import nullcontext
import pandas as pd
from dotenv import load_dotenv
from src.utils import load_csv, save_csv
from src.benchmark import BenchmarkRunner
from src.closeness_repair import ClosenessRepairRunner
from src.evaluator import (
    strict_match,
    ClosenessEvaluator,
    LLMClosenessEvaluator
)
from src.llm_backends.factory import get_llm


def get_active_venv():
    venv_name = os.path.basename(os.getenv("VIRTUAL_ENV", ""))
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

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Run benchmark for a specific model only (default: run all models for this venv)"
    )

    parser.add_argument(
        "--only-evaluate-closeness",
        action="store_true",
        help="Only evaluate closeness for an existing result CSV"
    )

    parser.add_argument(
        "--result-file",
        type=str,
        default=None,
        help="Path to an existing result CSV file for closeness-only evaluation"
    )

    args = parser.parse_args()

    load_dotenv()

    all_files = glob.glob("data/*.csv")

    df_list = []
    for file in all_files:
        print(f"Loading: {file}")
        df = load_csv(file)
        df_list.append(df)

    if df_list:
        df_all = pd.concat(df_list, ignore_index=True)
        print(f"Successfully combined {len(all_files)} files!")
    else:
        print("[ERROR] No CSV files found in data/ directory!")
        return

    df_all = df_all.fillna("")

    gpu_available = torch.cuda.is_available()

    closeness_eval = None
    prompt_rewriter_llm = None

    venv_name = get_active_venv()

    if gpu_available and venv_name != "venv_only_deepseek_vl2":
        print("[INFO] Loading Qwen3 once (Judge + Prompt Rewriter)")

        qwen_shared = get_llm(
            model_name="Qwen/Qwen3-4B-Instruct-2507",
            vision=False,
        )
        qwen_shared.load()

        closeness_eval = LLMClosenessEvaluator(qwen_shared)
        prompt_rewriter_llm = qwen_shared
    else:
        print("[INFO] Using string-based closeness evaluator (local fallback)")
        closeness_eval = ClosenessEvaluator()
        prompt_rewriter_llm = None

    if args.only_evaluate_closeness:
        if not args.result_file:
            print("[ERROR] --result-file is required when using --only-evaluate-closeness")
            return

        print(f"[INFO] Running closeness-only evaluation for: {args.result_file}")
        runner = ClosenessRepairRunner(closeness_eval)
        output_file = runner.run(args.result_file)
        print(f"[INFO] Wrote result file with refreshed closeness scores to: {output_file}")
        return

    if venv_name == "venv_only_deepseek_vl2":
        models_to_test = [
            {"name": "deepseek-ai/deepseek-vl2", "vision": True},
        ]
    elif venv_name in ["venv_all_other_models", "venv_all_other_models_py311"]:
        models_to_test = [
            {"name": "gpt2", "vision": False},
            {"name": "mistralai/Mistral-7B-Instruct-v0.3", "vision": False},
            {"name": "deepseek-v3.2", "vision": False},
            {"name": "DeepSeek-V3.1", "vision": False},
            {"name": "DeepSeek-V3", "vision": False},
            {"name": "DeepSeek-V2", "vision": False},
            {"name": "Salesforce/blip-image-captioning-base", "vision": True},
            {"name": "llava-hf/llava-onevision-qwen2-7b-ov-hf", "vision": True},
            {"name": "internlm/Intern-S1-mini", "vision": True},
            {"name": "claude-opus-4-6", "vision": True},
            {"name": "claude-3-opus-latest", "vision": True},
            {"name": "gpt-5.2", "vision": True},
            {"name": "gpt-4.1", "vision": True},
            {"name": "gpt-3.5-turbo", "vision": False},
            {"name": "Qwen/Qwen3-VL-235B-A22B-Instruct", "vision": True},
            {"name": "Qwen/Qwen2.5-VL-32B-Instruct", "vision": True},
            {"name": "Qwen/Qwen2-VL-2B-Instruct", "vision": True},
            {"name": "meta-llama/Llama-4-Scout-17B-16E-Instruct", "vision": True},
            {"name": "meta-llama/Llama-3.3-70B-Instruct", "vision": False},
            {"name": "meta-llama/Llama-3.1-70B-Instruct", "vision": False},
            {"name": "meta-llama/Meta-Llama-3-70B-Instruct", "vision": False},
            {"name": "google/gemma-3-27b-it", "vision": True},
            {"name": "google/gemma-2-9b-it", "vision": False},
            {"name": "gemini-3-pro-preview", "vision": True},
            {"name": "gemini-2.5-pro", "vision": True},
        ]
    else:
        print(f"[WARNING] Unknown virtual environment '{venv_name}'.")
        return

    if args.model:
        models_to_test = [m for m in models_to_test if m["name"] == args.model]
        if not models_to_test:
            print(f"[WARNING] Model '{args.model}' not found in this venv -> Cancelled")
            return
    else:
        print("[INFO] No model specified via arguments, so all models of the current venv will be run!")

    for model_info in models_to_test:
        model_name = model_info["name"]
        vision_enabled = model_info["vision"]

        print(f"\n=== Benchmarking {model_name} (Vision={vision_enabled}) ===")

        if vision_enabled:
            df = df_all.reset_index(drop=True)
            print(f"[INFO] Vision model -> {len(df)} total questions (text + image)")
        else:
            df = df_all[
                df_all["image_path"].isna()
                | (df_all["image_path"].str.strip() == "")
            ].reset_index(drop=True)
            print(f"[INFO] Text model -> {len(df)} text questions")

        if df.empty:
            print("[WARNING] No suitable questions found -> skipping model")
            continue

        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if gpu_available else
            nullcontext()
        )

        with autocast_context:
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
                vision=vision_enabled,
                prompt_rewriter_llm=prompt_rewriter_llm
            )

            if args.experiment in ["one-shot", "all"]:
                print(f"--- {model_name} | One-Shot ---")
                one_shot_path = f"results/{model_name.replace('/', '_')}_one_shot.csv"
                one_shot_df = runner.run_one_shot(output_path=one_shot_path)
                save_csv(one_shot_df, one_shot_path)

            if args.experiment in ["two-shot", "all"]:
                print(f"--- {model_name} | Two-Shot ---")
                two_shot_path = f"results/{model_name.replace('/', '_')}_two_shot.csv"
                two_shot_df = runner.run_two_shot(output_path=two_shot_path)
                save_csv(two_shot_df, two_shot_path)

            if args.experiment in ["lfe", "all"]:
                print(f"--- {model_name} | Learning-from-Experience ---")
                lfe_path = f"results/{model_name.replace('/', '_')}_lfe.csv"
                lfe_df = runner.run_learning_from_experience(output_path=lfe_path)
                save_csv(lfe_df, lfe_path)


if __name__ == "__main__":
    main()
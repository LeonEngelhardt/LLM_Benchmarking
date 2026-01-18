from utils import load_csv, save_csv
from llm_models import HuggingFaceMultimodalLLM
from benchmark import BenchmarkRunner
from evaluator import ClosenessEvaluator, strict_match

if __name__ == "__main__":
    df = load_csv("data/dataset.csv")
    closeness_eval = ClosenessEvaluator()

    # LLaMa-2
    llama_model = HuggingFaceMultimodalLLM("meta-llama/Llama-2-7b-chat-hf")
    llama_model.load_model()

    runner_llama = BenchmarkRunner(
        df,
        llama_model,
        evaluator=strict_match,
        closeness_evaluator=closeness_eval
    )

    llama_one_shot = runner_llama.run_one_shot()
    save_csv(llama_one_shot, "results/llama_one_shot_results.csv")

    llama_two_shot = runner_llama.run_two_shot()
    save_csv(llama_two_shot, "results/llama_two_shot_results.csv")

    llama_lfe = runner_llama.run_learning_from_experience()
    save_csv(llama_lfe, "results/llama_learning_from_experience_results.csv")

    # MPT/MosaicML
    print("Starte Benchmark für MPT/MosaicML...")
    mpt_model = HuggingFaceMultimodalLLM("mosaicml/mpt-7b-instruct")
    mpt_model.load_model()

    runner_mpt = BenchmarkRunner(
        df,
        mpt_model,
        evaluator=strict_match,
        closeness_evaluator=closeness_eval
    )

    mpt_one_shot = runner_mpt.run_one_shot()
    save_csv(mpt_one_shot, "results/mpt_one_shot_results.csv")

    mpt_two_shot = runner_mpt.run_two_shot()
    save_csv(mpt_two_shot, "results/mpt_two_shot_results.csv")

    mpt_lfe = runner_mpt.run_learning_from_experience()
    save_csv(mpt_lfe, "results/mpt_learning_from_experience_results.csv")
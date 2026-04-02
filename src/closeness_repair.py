import os
import pandas as pd
from tqdm import tqdm

from src.utils import load_csv, save_csv


class ClosenessRepairRunner:
    def __init__(self, closeness_evaluator):
        self.closeness_evaluator = closeness_evaluator

    def run(self, result_file: str) -> str:
        df = load_csv(result_file).fillna("")

        required_columns = {"ground_truth", "llm_answer"}
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(
                f"Result file is missing required columns: {sorted(missing)}"
            )

        closeness_scores = []
        for _, row in tqdm(
            df.iterrows(),
            total=len(df),
            desc="Closeness-Only"
        ):
            score = self.closeness_evaluator.score(
                str(row["llm_answer"]),
                str(row["ground_truth"])
            )
            closeness_scores.append(score)

        df["closeness_score"] = closeness_scores

        base, ext = os.path.splitext(result_file)
        output_file = f"{base}_with_closeness{ext}"
        save_csv(df, output_file)
        return output_file
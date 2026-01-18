from tqdm import tqdm
from src.prompt import build_prompt
from src.evaluator import strict_match
import pandas as pd

class BenchmarkRunner:
    def __init__(self, df: pd.DataFrame, llm, evaluator=strict_match, closeness_evaluator=None):
        self.df = df
        self.llm = llm
        self.evaluator = evaluator
        self.closeness_evaluator = closeness_evaluator

    def _evaluate_answer(self, pred, truth):
        is_correct = self.evaluator(pred, truth)
        closeness = None
        if self.closeness_evaluator:
            closeness = self.closeness_evaluator.score(pred, truth)
        return is_correct, closeness

    def run_one_shot(self):
        results = []
        for _, row in tqdm(self.df.iterrows(), total=len(self.df)):
            example_rows = self.df.sample(2).to_dict('records')
            for ex in example_rows:
                prompt = build_prompt(row['question'], [ex], mode="one_shot")
                answer = self.llm.generate(prompt)
                is_correct, closeness = self._evaluate_answer(answer, row['answer'])
                results.append({
                    "mode": "one_shot",
                    "question": row['question'],
                    "ground_truth": row['answer'],
                    "llm_answer": answer,
                    "is_correct": is_correct,
                    "closeness_score": closeness
                })
        return pd.DataFrame(results)

    def run_two_shot(self):
        results = []
        for _, row in tqdm(self.df.iterrows(), total=len(self.df)):
            examples = self.df.sample(2).to_dict('records')
            prompt = build_prompt(row['question'], examples, mode="two_shot")
            answer = self.llm.generate(prompt)
            is_correct, closeness = self._evaluate_answer(answer, row['answer'])
            results.append({
                "mode": "two_shot",
                "question": row['question'],
                "ground_truth": row['answer'],
                "llm_answer": answer,
                "is_correct": is_correct,
                "closeness_score": closeness
            })
        return pd.DataFrame(results)

    def run_learning_from_experience(self):
        results = []
        for _, row in tqdm(self.df.iterrows(), total=len(self.df)):
            examples = self.df.sample(2).to_dict('records')
            prompt = build_prompt(row['question'], examples, mode="learning_from_experience")
            answer = self.llm.generate(prompt)
            is_correct = answer.strip().lower() == "you are correct"
            closeness = None
            if self.closeness_evaluator:
                closeness = self.closeness_evaluator.score(answer, row['answer'])
            results.append({
                "mode": "learning_from_experience",
                "question": row['question'],
                "ground_truth": row['answer'],
                "llm_answer": answer,
                "is_correct": is_correct,
                "closeness_score": closeness
            })
        return pd.DataFrame(results)

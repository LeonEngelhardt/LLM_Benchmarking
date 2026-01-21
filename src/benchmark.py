from tqdm import tqdm
from src.prompt import build_prompt
from src.evaluator import strict_match
import pandas as pd

class BenchmarkRunner:
    def __init__(self, df: pd.DataFrame, llm, evaluator=strict_match, closeness_evaluator=None, vision=False):
        self.df = df
        self.llm = llm
        self.evaluator = evaluator
        self.closeness_evaluator = closeness_evaluator
        self.vision = vision

        if vision:
            self.df_targets = df[df['image_path'].notna()].reset_index(drop=True)
        else:
            self.df_targets = df[df['image_path'].isna()].reset_index(drop=True)
        self.df_examples = df[df['is_added'].fillna(False) == True].reset_index(drop=True)

    def _evaluate_answer(self, pred, truth):
        is_correct = self.evaluator(pred, truth)
        closeness = None
        if self.closeness_evaluator:
            closeness = self.closeness_evaluator.score(pred, truth)
        return is_correct, closeness

    def _get_image_path(self, row):
        img = row.get('image_path')
        if isinstance(img, str) and img.strip():
            return img
        return None

    def _get_example_rows(self, target_row, n=2):
        example_rows = self.df_examples[self.df_examples['id_original_question'] == target_row['id']] \
                        .sort_values('example_number').to_dict('records')
        return example_rows[:n]

    def run_one_shot(self):
        results = []
        for _, row in tqdm(self.df_targets.iterrows(), total=len(self.df_targets), desc="One-Shot"):
            example_rows = self._get_example_rows(row, n=None)
            for ex in example_rows:
                prompt = build_prompt(row, [ex], mode="one_shot", vision=self.vision)

                image_path = self._get_image_path(row) if self.vision else None

                print("******************************************************************")
                print(prompt)
                print("******************************************************************")

                if self.vision and image_path:
                    answer = self.llm.generate(prompt, image_path=image_path)
                else:
                    answer = self.llm.generate(prompt)

                if not answer:
                    answer = "[No answer]"
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
        for _, row in tqdm(self.df_targets.iterrows(), total=len(self.df_targets), desc="Two-Shot"):
            example_rows = self._get_example_rows(row, n=2)
            if not example_rows:
                continue

            prompt = build_prompt(row, example_rows, mode="two_shot", vision=self.vision)
            print("===== Two-Shot Prompt =====")
            print(prompt)
            print("===========================")

            image_path = self._get_image_path(row) if self.vision else None
            answer = self.llm.generate(prompt, image_path=image_path) if image_path else self.llm.generate(prompt)

            if not answer:
                answer = "[No answer]"

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
        for _, row in tqdm(self.df_targets.iterrows(), total=len(self.df_targets), desc="Learning-from-Experience"):
            example_rows = self._get_example_rows(row, n=2)
            if not example_rows:
                continue

            prompt = build_prompt(row, example_rows, mode="learning_from_experience", vision=self.vision)
            print("===== LfE Prompt =====")
            print(prompt)
            print("======================")

            image_path = self._get_image_path(row) if self.vision else None
            answer = self.llm.generate(prompt, image_path=image_path) if image_path else self.llm.generate(prompt)

            if not answer:
                answer = "[No answer]"

            is_correct = answer.strip().lower() == "you are correct"
            closeness = self.closeness_evaluator.score(answer, row['answer']) if self.closeness_evaluator else None

            results.append({
                "mode": "learning_from_experience",
                "question": row['question'],
                "ground_truth": row['answer'],
                "llm_answer": answer,
                "is_correct": is_correct,
                "closeness_score": closeness
            })
        return pd.DataFrame(results)
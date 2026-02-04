from tqdm import tqdm
from src.prompt import build_prompt
from src.evaluator import strict_match
import pandas as pd
from PIL import Image
import torch

class BenchmarkRunner:
    def __init__(self, df: pd.DataFrame, llm, evaluator=strict_match, closeness_evaluator=None, vision=False):
        self.df = df
        self.llm = llm
        self.evaluator = evaluator
        self.closeness_evaluator = closeness_evaluator
        self.vision = vision

        if vision:
            self.df_targets = df[df['image_path'].notna() & (df['image_path'].str.strip() != "")].reset_index(drop=True)
        else:
            self.df_targets = df[df['image_path'].isna() | (df['image_path'].str.strip() == "")].reset_index(drop=True)
        self.df_examples = df[df['is_added'].fillna(False) == True].reset_index(drop=True)
        self.df_originals = df[df['is_added'].fillna(False) == False].reset_index(drop=True)

    def _extract_final_answer(self, text: str) -> str:
        separator = "Answer:"
        if separator in text:
            answer = text.split(separator)[-1].strip()
            if answer.endswith("."):
                answer = answer[:-1]
            return answer.strip()
        return text.strip()

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

    def _get_example_rows(self, target_row, n=None):
        example_rows = self.df_examples[self.df_examples['id_original_question'] == target_row['id']] \
            .sort_values('example_number').to_dict('records')
        if n is not None:
            return example_rows[:n]
        return example_rows

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
                    raw_answer = self.llm.generate(prompt, image_path=image_path)
                else:
                    raw_answer = self.llm.generate(prompt)

                if not raw_answer:
                    raw_answer = "[No answer]"
                
                clean_answer = self._extract_final_answer(raw_answer)

                is_correct, closeness = self._evaluate_answer(clean_answer, row['answer'])
                results.append({
                    "mode": "one_shot",
                    "question": row['question'],
                    "ground_truth": row['answer'],
                    "llm_raw_output": raw_answer,
                    "llm_answer": clean_answer,
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
            if self.vision and image_path:
                raw_answer = self.llm.generate(prompt, image_path=image_path)
            else:
                raw_answer = self.llm.generate(prompt)

            if not raw_answer:
                raw_answer = "[No answer]"

            clean_answer = self._extract_final_answer(raw_answer)

            is_correct, closeness = self._evaluate_answer(clean_answer, row['answer'])
            results.append({
                "mode": "two_shot",
                "question": row['question'],
                "ground_truth": row['answer'],
                "llm_raw_output": raw_answer,
                "llm_answer": clean_answer,
                "is_correct": is_correct,
                "closeness_score": closeness
            })
        return pd.DataFrame(results)

    def run_learning_from_experience(self, max_iterations=2):
        results = []

        for _, row in tqdm(self.df_originals.iterrows(),
                        total=len(self.df_originals),
                        desc="Learning-from-Experience"):

            image_path = self._get_image_path(row) if self.vision else None

            current_prompt = build_prompt(row, [], mode="zero_shot", vision=self.vision)
            if not current_prompt or not current_prompt.strip():
                # Ensure fallback also asks for Answer:
                current_prompt = f"Question: {row['question']}\nAnswer:"

            final_raw_answer = None
            final_clean_answer = None
            num_iterations = 0
            is_correct = False
            closeness = None

            for _ in range(max_iterations):
                num_iterations += 1

                if self.vision and image_path:
                    raw_answer = self.llm.generate(current_prompt, image_path=image_path)
                else:
                    raw_answer = self.llm.generate(current_prompt)

                if not raw_answer:
                    raw_answer = "[No answer]"
                
                clean_answer = self._extract_final_answer(raw_answer)

                is_correct, closeness = self._evaluate_answer(clean_answer, row['answer'])

                final_raw_answer = raw_answer
                final_clean_answer = clean_answer

                if is_correct:
                    break

                current_prompt += "\n\nYour answer was incorrect. Please try again."

            results.append({
                "mode": "learning_from_experience",
                "question": row['question'],
                "ground_truth": row['answer'],
                "llm_raw_output": final_raw_answer,
                "llm_answer": final_clean_answer,
                "is_correct": is_correct,
                "num_iterations": num_iterations,
                "closeness_score": closeness
            })

        return pd.DataFrame(results)
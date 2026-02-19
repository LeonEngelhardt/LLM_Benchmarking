import pandas as pd
from tqdm import tqdm
from src.prompt import build_prompt_parts #,build_prompt
from src.evaluator import strict_match
from src.utils import normalize_image_path
from transformers import AutoTokenizer

class BenchmarkRunner:
    def __init__(self, df: pd.DataFrame, llm, evaluator=strict_match, closeness_evaluator=None, vision=False):
        self.df = df
        self.llm = llm
        self.evaluator = evaluator
        self.closeness_evaluator = closeness_evaluator
        self.vision = vision
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

        self.df_examples = df[df['is_added'].fillna(False) == True].reset_index(drop=True)
        self.df_originals = df[df['is_added'].fillna(False) == False].reset_index(drop=True)

        if vision:
            # VLMs receive both text-only and image questions
            self.df_targets = self.df_originals.copy().reset_index(drop=True)
        else:
            # Text-only models only receive questions without images
            self.df_targets = self.df_originals[self.df_originals['image_path'].isna() | (self.df_originals['image_path'].str.strip() == "")].reset_index(drop=True)

        # Special test case for BLIP only --> local testing
        if self.llm.__class__.__name__ == "BlipLLM":
            self.df_targets = self.df_originals[self.df_originals['image_path'].notna() & (self.df_originals['image_path'].str.strip() != "")].reset_index(drop=True)
            self.df_examples = self.df_examples[self.df_examples['image_path'].notna() & (self.df_examples['image_path'].str.strip() != "")].reset_index(drop=True)
            self.df_originals = self.df_originals[self.df_originals['image_path'].notna() & (self.df_originals['image_path'].str.strip() != "")].reset_index(drop=True)

        print("EXXXXXAMPLES: ", self.df_examples)
        print("OOOOOOORRRRRRRRRRIGINALES: ", self.df_originals)
        print("TTTTTTTTTTTTTAAAAAAAAAAAARGETS: ", self.df_targets)
    
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
            return normalize_image_path(img)
        return None

    def _get_example_rows(self, target_row, n=None):
        example_rows = self.df_examples[self.df_examples['id_original_question'] == target_row['id']] \
            .sort_values('example_number').to_dict('records')
        if n is not None:
            return example_rows[:n]
        return example_rows

    def run_one_shot(self):
        results = []

        for _, row in tqdm(
            self.df_targets.iterrows(),
            total=len(self.df_targets),
            desc="One-Shot"
        ):

            example_rows = self._get_example_rows(row, n=None)

            for ex in example_rows:
                prompt_parts = build_prompt_parts(row, [ex], mode="one_shot")

                prompt_parts = self.check_token_length(prompt_parts)

                image_paths = []

                if self.vision:
                    ex_img = self._get_image_path(ex)
                    if ex_img:
                        image_paths.append(ex_img)

                    target_img = self._get_image_path(row)
                    if target_img:
                        image_paths.append(target_img)

                #raw_answer = self.llm.generate(prompt_parts, image_paths=image_paths if self.vision else None)
                raw_answer = self.llm.generate(prompt_parts)

                if not raw_answer:
                    raw_answer = "[No answer]"

                clean_answer = self._extract_final_answer(raw_answer)
                is_correct, closeness = self._evaluate_answer(
                    clean_answer,
                    row["answer"]
                )

                results.append({
                    "mode": "one_shot",
                    "question": row["question"],
                    "ground_truth": row["answer"],
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
            if len(example_rows) < 2:
                continue

            prompt_parts = build_prompt_parts(row, example_rows, mode="two_shot")

            prompt_parts = self.check_token_length(prompt_parts)

            image_paths = []

            if self.vision:
                for ex in example_rows:
                    img = self._get_image_path(ex)
                    if img:
                        image_paths.append(img)

                target_img = self._get_image_path(row)
                if target_img:
                    image_paths.append(target_img)

            #raw_answer = self.llm.generate(
            #    prompt_parts,
            #    image_paths=image_paths if self.vision else None)
            raw_answer = self.llm.generate(prompt_parts)


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
    
    def run_learning_from_experience(self):
        results = []

        for _, row in tqdm(self.df_originals.iterrows(),
                        total=len(self.df_originals),
                        desc="Learning-from-Experience"):

            example_rows = self._get_example_rows(row, n=2)

            instruction, _ = build_prompt_parts(row, [], mode="zero_shot")

            accumulated_blocks = []
            accumulated_image_paths = []

            for example in example_rows:

                ex_instruction, ex_blocks = build_prompt_parts(
                    example,
                    [],
                    mode="zero_shot"
                )

                accumulated_blocks.extend(ex_blocks)

                if self.vision:
                    img = self._get_image_path(example)
                    if img:
                        accumulated_image_paths.append(img)

                prompt_parts = (instruction, accumulated_blocks)

                prompt_parts = self.check_token_length(prompt_parts)

                raw_answer = self.llm.generate(
                    prompt_parts,
                    image_paths=accumulated_image_paths if self.vision else None
                )

                if not raw_answer:
                    raw_answer = "[No answer]"

                clean_answer = self._extract_final_answer(raw_answer)

                is_correct, _ = self._evaluate_answer(
                    clean_answer,
                    example["answer"]
                )

                if is_correct:
                    feedback = (
                        "The previous answer was CORRECT.\n"
                        f"The previous answer was: {clean_answer}\n\n"
                    )
                else:
                    feedback = (
                        "The previous answer was INCORRECT.\n"
                        f"The previous answer was: {clean_answer}\n\n"
                    )

                accumulated_blocks.append({
                    "type": "text",
                    "text": feedback
                })


            _, target_blocks = build_prompt_parts(
                row,
                [],
                mode="zero_shot"
            )

            accumulated_blocks.extend(target_blocks)

            if self.vision:
                img = self._get_image_path(row)
                if img:
                    accumulated_image_paths.append(img)

            final_prompt_parts = (instruction, accumulated_blocks)

            final_raw_answer = self.llm.generate(
                final_prompt_parts,
                image_paths=accumulated_image_paths if self.vision else None
            )

            if not final_raw_answer:
                final_raw_answer = "[No answer]"

            clean_answer = self._extract_final_answer(final_raw_answer)

            is_correct, closeness = self._evaluate_answer(
                clean_answer,
                row["answer"]
            )

            results.append({
                "mode": "learning_from_experience",
                "question": row["question"],
                "ground_truth": row["answer"],
                "llm_raw_output": final_raw_answer,
                "llm_answer": clean_answer,
                "is_correct": is_correct,
                "num_iterations": 3,
                "closeness_score": closeness
            })

        return pd.DataFrame(results)
    

    def check_token_length(self, prompt_parts):
        # Tokenlänge des Prompts berechnen
        prompt_text = " ".join([part['text'] for part in prompt_parts if 'text' in part])
        tokens = self.tokenizer.encode(prompt_text)
        
        # Wenn die Länge der Tokens zu groß ist, verwende Qwen3, um das Prompt zu kürzen
        if len(tokens) > 4096:
            print(f"Prompt überschreitet 4096 Tokens, Kürzung erforderlich!")
            # Hier verwendest du Qwen3 oder eine andere Methode, um das Prompt zu kürzen.
            prompt_parts = self.shorten_prompt_with_qwen3(prompt_parts)
        
        return prompt_parts
    

    def shorten_prompt_with_qwen3(self, prompt_parts):
        """
        Verwendet Qwen3, um das Prompt zu kürzen, ohne die Struktur oder wichtige Informationen zu verlieren.
        """
        # Qwen3 verwenden (z.B. über die ClosenessEvaluator oder eine andere Methode)
        print("Verwende Qwen3, um das Prompt zu kürzen...")
        prompt_text = " ".join([part['text'] for part in prompt_parts if 'text' in part])
        shortened_prompt = self.closeness_evaluator.llm.generate(prompt_text)  # Qwen3 kürzt das Prompt

        # Kürzungslogik: Prompt verkürzen, aber die Struktur beibehalten
        # Hier ist der Ansatz abhängig von der Qwen3-Methode, die du verwendest
        # Du könntest auch die wichtigen Teile des Prompts priorisieren und nur unwichtige Teile entfernen

        # Beispiel: Einfache Kürzung, die die Struktur beibehält
        return [{"type": "text", "text": shortened_prompt}]







    """def run_learning_from_experience(self, max_iterations):
        results = []

        for _, row in tqdm(self.df_originals.iterrows(),
                        total=len(self.df_originals),
                        desc="Learning-from-Experience"):

            example_rows = []
            prompt_parts = build_prompt_parts(row, example_rows, mode="zero_shot")

            image_paths = []

            if self.vision:
                target_img = self._get_image_path(row)
                if target_img:
                    image_paths.append(target_img)

            current_prompt_parts = prompt_parts

            final_raw_answer = None
            num_iterations = 0
            is_correct = False
            closeness = None

            for _ in range(max_iterations):
                num_iterations += 1

                #raw_answer = self.llm.generate(
                #    current_prompt_parts,
                #    image_paths=image_paths if self.vision else None)
                raw_answer = self.llm.generate(prompt_parts)


                if not raw_answer:
                    raw_answer = "[No answer]"

                clean_answer = self._extract_final_answer(raw_answer)
                final_raw_answer = raw_answer

                is_correct, closeness = self._evaluate_answer(clean_answer, row['answer'])

                if is_correct:
                    break

                #current_prompt_parts["target"] += "\nYour answer was incorrect. Try again.\nAnswer:"
                for block in current_prompt_parts:
                    if isinstance(block, dict) and block.get("type") == "text":
                        if block["text"].startswith("Now answer the following question"):
                            block["text"] += "\nYour answer was incorrect. Try again.\nAnswer:"
                            break

            results.append({
                "mode": "learning_from_experience",
                "question": row['question'],
                "ground_truth": row['answer'],
                "llm_raw_output": final_raw_answer,
                "llm_answer": clean_answer,
                "is_correct": is_correct,
                "num_iterations": num_iterations,
                "closeness_score": closeness
            })

        return pd.DataFrame(results)"""
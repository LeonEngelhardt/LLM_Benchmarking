from sentence_transformers import SentenceTransformer, util
import re

def strict_match(pred: str, truth: str) -> bool:
    return pred.strip().lower() == truth.strip().lower()

class LLMClosenessEvaluator:
    def __init__(self, llm):
        self.llm = llm

    def score(self, pred: str, truth: str) -> float:
        prompt = f"""
How would you rank the similarity between the following two answers based on a scale from 0 to 10?

Answer 1:
{pred}

Answer 2:
{truth}

Scoring Criteria:
        - 10: The prediction is factually identical to the ground truth.
        - 0: The prediction is completely wrong or irrelevant.
        - Use intermediate numbers for partially correct answers.

        Output ONLY a single number between 0 and 10.
"""

        raw = self.llm.generate(prompt)

        if not raw:
            return None

        match = re.search(r"(\d+(\.\d+)?)", raw)
        if not match:
            return None

        value = float(match.group(1))
        return max(0.0, min(10.0, value))

# this evaluator is only here to be able to test the framework locally! -> QWEN will rate the answer from 0 to 10 when on a cluster
class ClosenessEvaluator:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def score(self, pred: str, truth: str) -> float:
        emb_pred = self.model.encode(pred, convert_to_tensor=True)
        emb_truth = self.model.encode(truth, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(emb_pred, emb_truth).item()
        closeness = max(1.0, min(10.0, ((similarity + 1) / 2) * 9 + 1))
        return closeness
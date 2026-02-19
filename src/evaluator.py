from sentence_transformers import SentenceTransformer, util
import re

def strict_match(pred: str, truth: str) -> bool:
    return pred.strip().lower() == truth.strip().lower()

class LLMClosenessEvaluator:
    def __init__(self, llm):
        self.llm = llm

    def score(self, pred: str, truth: str) -> float:
        prompt = f"""
You are an expert evaluator.

Your task is to score how close the PREDICTED answer is to the GROUND TRUTH answer.

Evaluate strictly based on factual correctness and completeness.

PREDICTED ANSWER:
\"\"\"{pred}\"\"\"

GROUND TRUTH:
\"\"\"{truth}\"\"\"

Scoring Rules (0-10 scale):

10 = Factually identical. Same meaning, no missing or incorrect information.
9  = Same meaning, extremely minor wording differences only.
7-8 = Mostly correct, but missing small details OR contains minor inaccuracies.
4-6 = Partially correct. Contains correct elements but important details are missing or incorrect.
1-3 = Mostly incorrect. Only small fragments are correct.
0  = Completely wrong, contradictory, or irrelevant.

Important:
- Do NOT be generous.
- Penalize missing information.
- Penalize incorrect facts.
- Do NOT explain your reasoning.
- Output ONLY a single number between 0 and 10.
- Output nothing else.

Score:
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
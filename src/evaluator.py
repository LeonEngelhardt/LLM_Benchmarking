from sentence_transformers import SentenceTransformer, util

def strict_match(pred: str, truth: str) -> bool:
    return pred.strip().lower() == truth.strip().lower()

class ClosenessEvaluator:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def score(self, pred: str, truth: str) -> float:
        emb_pred = self.model.encode(pred, convert_to_tensor=True)
        emb_truth = self.model.encode(truth, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(emb_pred, emb_truth).item()
        closeness = max(1.0, min(10.0, ((similarity + 1) / 2) * 9 + 1))
        return closeness

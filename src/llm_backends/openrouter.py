import os, requests
from .base import BaseLLM

class OpenRouterLLM(BaseLLM):
    def __init__(self, model_name):
        super().__init__(model_name)

    def load(self): pass

    def generate(self, prompt, image_path=None):
        res = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            },
            json={
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        return res.json()["choices"][0]["message"]["content"]

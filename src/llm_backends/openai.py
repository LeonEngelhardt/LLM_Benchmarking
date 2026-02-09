import os
from openai import OpenAI
from .base import BaseLLM

class OpenAILLM(BaseLLM):
    def __init__(self, model_name):
        super().__init__(model_name)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def load(self): pass

    def generate(self, prompt, image_path=None):
        res = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return res.choices[0].message.content
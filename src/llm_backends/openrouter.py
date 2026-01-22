import os
from openai import OpenAI
from .base import BaseLLM

class OpenRouterBackend(BaseLLM):

    def __init__(self, model_name, vision=False, verbose=False):
        super().__init__(model_name, vision, verbose)
        self.client = OpenAI(
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )

    def generate(self, prompt, image_path=None):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role":"user","content":prompt}],
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[OpenRouter Error] {str(e)}"
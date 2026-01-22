import os
from openai import OpenAI
from .base import BaseLLM

class OpenAIBackend(BaseLLM):
    def __init__(self, model_name, vision=False, verbose=False):
        super().__init__(model_name, vision, verbose)
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def generate(self, prompt, image_path=None):
        try:
            if self.vision and image_path:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role":"user","content":prompt}],
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role":"user","content":prompt}],
                )
            return response.choices[0].message.content
        except Exception as e:
            return f"[OpenAI Error] {str(e)}"
import os
import anthropic
from .base import BaseLLM

class AnthropicLLM(BaseLLM):
    def __init__(self, model_name):
        super().__init__(model_name)
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def load(self): pass

    def generate(self, prompt, image_path=None):
        msg = self.client.messages.create(
            model=self.model_name,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        )
        return msg.content[0].text
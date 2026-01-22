import os
import anthropic
from .base import BaseLLM

class AnthropicBackend(BaseLLM):
    def __init__(self, model_name, vision=False, verbose=False):
        super().__init__(model_name, vision, verbose)
        self.client = anthropic.Client(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def generate(self, prompt, image_path=None):
        try:
            response = self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=4000,
                stop_sequences=["\n\n"]
            )
            return response.completion
        except Exception as e:
            return f"[Anthropic Error] {str(e)}"
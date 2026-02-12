import os
from openai import OpenAI
from .base import BaseLLM

class DeepSeekV3LLM(BaseLLM):
    def __init__(self, model_name, vision):
        super().__init__(model_name, vision)
        self.temperature = 1.0
        self.top_p = 1.0
        self.loaded = False

    def load(self):
        print(f"Initializing DeepSeek model '{self.model_name}'...")

        self.client = OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )

        self.loaded = True
        print("DeepSeek client initialized.")

    def generate(self, prompt):
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call `load()` first.")

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=1024,
            stream=False,
        )

        return response.choices[0].message.content.strip()
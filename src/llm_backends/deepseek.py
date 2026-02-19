import os
from openai import OpenAI
from .base import BaseLLM

class DeepSeekV3LLM(BaseLLM):
    def __init__(self, model_name, vision=False):
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

    def generate(self, prompt_parts, examples=None):
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call `load()` first.")

        if isinstance(prompt_parts, list) or isinstance(prompt_parts, tuple):
            instruction, blocks = prompt_parts
            text_blocks = [p["text"] for p in blocks if p.get("type") == "text"]
            full_prompt = "\n".join(text_blocks)
        else:
            instruction = ""
            full_prompt = str(prompt_parts)

        if examples:
            full_prompt = "\n".join(examples) + "\n" + full_prompt

        print("Instruction:", instruction)
        print("Prompt:", full_prompt)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": full_prompt},
            ],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=1024,
            stream=False,
        )

        return response.choices[0].message.content.strip()







"""import os
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

        return response.choices[0].message.content.strip()"""
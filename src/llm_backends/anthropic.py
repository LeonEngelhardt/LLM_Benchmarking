import os
import anthropic
from .base import BaseLLM

class AnthropicLLM(BaseLLM):
    def __init__(self, model_name):
        super().__init__(model_name, vision=True)
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.loaded = False

    def load(self):
        self.loaded = True
        print(f"Anthropic model '{self.model_name}' loaded successfully.")

    def generate(self, prompt_parts, max_tokens=512):
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call `load()` first.")

        if not isinstance(prompt_parts, tuple) or len(prompt_parts) != 2:
            raise ValueError("prompt_parts must be a tuple: (instruction, blocks)")

        instruction, blocks = prompt_parts

        messages = []
        if instruction:
            messages.append({"role": "system", "content": instruction})

        user_content = []
        for block in blocks:
            if block["type"] == "text":
                user_content.append({"type": "text", "text": block["text"]})
            elif block["type"] == "image":
                source = block.get("source", {})
                user_content.append({
                    "type": "image",
                    "source": {"type": "url", "url": source.get("url")}
                })

        messages.append({"role": "user", "content": user_content})

        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            messages=messages
        )

        response_text = ""
        for block in response.content:
            if block.type == "text":
                response_text += block.text

        if "Answer:" in response_text:
            return response_text.split("Answer:")[-1].strip()

        return response_text.strip()





"""import os
import anthropic
from .base import BaseLLM

class AnthropicLLM(BaseLLM):
    def __init__(self, model_name):
        super().__init__(model_name, vision=True)
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.loaded = False

    def load(self):
        self.loaded = True
        print(f"Anthropic model '{self.model_name}' loaded successfully.")

    def generate(self, prompt_parts, max_tokens=512):
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call `load()` first.")

        if isinstance(prompt_parts, str):
            prompt_parts = [{"type": "text", "text": prompt_parts}]

        message_content = []
        for block in prompt_parts:
            if block["type"] == "text":
                message_content.append({"type": "text", "text": block["text"]})
            elif block["type"] == "image":
                source = block.get("source", {})
                message_content.append({
                    "type": "image",
                    "source": {"type": "url", "url": source.get("url")}
                })

        print(message_content)
        
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            system=self.system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": message_content
                }
            ]
        )

        response_text = ""
        for block in response.content:
            if block.type == "text":
                response_text += block.text

        if "Answer:" in response_text:
            return response_text.split("Answer:")[-1].strip()

        return response_text.strip()"""
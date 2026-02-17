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

        return response_text.strip()



"""import os
import anthropic
from .base import BaseLLM

class AnthropicLLM(BaseLLM):
    def __init__(self, model_name):
        super().__init__(model_name)
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def load(self):
        self.loaded = True
        print("Model loaded successfully.")

    def generate(self, prompt, image_path):
        content_blocks = []

        if image_path:

            # in case we need to provide an image list when we have two shot 
            if isinstance(image_path, list):
                for i, img_url in enumerate(image_path):
                    content_blocks.append({
                        "type": "text",
                        "text": f"Image {i+1}:"
                    })
                    content_blocks.append({
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": img_url
                        }
                    })
            else:
                content_blocks.append({
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": image_path
                    }
                })

        content_blocks.append({
            "type": "text",
            "text": prompt
        })

        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=512,
            system=self.system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": content_blocks
                }
            ]
        )

        response_text = ""
        for block in message.content:
            if block.type == "text":
                response_text += block.text

        if "Answer:" in response_text:
            return response_text.split("Answer:")[-1].strip()

        return response_text.strip()"""
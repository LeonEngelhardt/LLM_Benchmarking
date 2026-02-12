import os
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

        return response_text.strip()
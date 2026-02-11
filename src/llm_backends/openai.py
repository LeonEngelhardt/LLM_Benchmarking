import os
from openai import OpenAI
from .base import BaseLLM

class OpenAILLM(BaseLLM):
    def __init__(self, model_name):
        super().__init__(model_name)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def load(self): self.loaded = True
    
    def generate(self, prompt, image_path=None):   
        # In case we use the non-vision API
        #response = self.client.responses.create(
        #    model=self.model_name,
        #    instructions="",
        #    input=prompt,
        #)
  
        if image_path:
            input_content = [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": image_path},
                    ],
                }
            ]
        else:
            input_content = prompt

        response = self.client.responses.create(
            model=self.model_name,
            input=input_content,
            max_output_tokens=256,
            temperature=0.0
        )

        text = response.output_text.strip()

        if "Answer:" in text:
            return text.split("Answer:")[-1].strip()

        return text
import os
from openai import OpenAI
from .base import BaseLLM

class OpenAILLM(BaseLLM):
    def __init__(self, model_name):
        super().__init__(model_name)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def load(self): self.loaded = True # pass

    def generate(self, prompt, image_path=None):
        res = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.7
        )
        # return res.choices[0].message.content

        # Postprocessing of answer --> if not reimplement return statement above...
        text = res.choices[0].message.content.strip()

        if "Answer:" in text:
            return text.split("Answer:")[-1].strip()

        return text
    

    # if we use gpt4o for text and images:
    """def generate(self, prompt, image_path=None):
    if image_path:
        # structured multimodal payload
        messages = [
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": prompt },
                    { "type": "image_url",
                      "image_url": { "url": image_path }
                    }
                ]
            }
        ]
    else:
        messages = [{"role": "user", "content": prompt}]
    
    res = self.client.chat.completions.create(
        model=self.model_name,
        messages=messages,
        max_tokens=256,
        temperature=0.7
    )""" 
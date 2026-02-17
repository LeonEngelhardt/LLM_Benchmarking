from openai import OpenAI
from .base import BaseLLM
import os


class OpenAILLM(BaseLLM):

    def __init__(self, model_name):
        pass
        #super().__init__(model_name)
        #self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def load(self):
        pass
        #self.loaded = True

    def generate(self, prompt_parts):

        if not isinstance(prompt_parts, list):
            raise ValueError("prompt_parts must be a list of content blocks")
        
        print(prompt_parts)

        #response = self.client.responses.create(
        #    model=self.model_name,
        #    input=[{
        #        "role": "user",
        #        "content": prompt_parts
        #    }],
        #    max_output_tokens=512,
        #    temperature=0.0
        #)

        #text = response.output_text.strip()

        #return text



"""import os
from openai import OpenAI
from .base import BaseLLM


class OpenAILLM(BaseLLM):
    def __init__(self, model_name):
        pass
        #super().__init__(model_name)
        #self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def load(self):
        pass
        #self.loaded = True

    def generate(self, prompt_parts, image_paths=None):

        content_blocks = []

        content_blocks.append({
            "type": "text",
            "text": prompt_parts["instruction"]
        })

        if image_paths is None:
            image_paths = []
        elif isinstance(image_paths, str):
            image_paths = [image_paths]

        image_index = 0

        for example_text in prompt_parts["examples"]:
            content_blocks.append({
                "type": "text",
                "text": example_text
            })

            if image_index < len(image_paths):
                content_blocks.append({
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": image_paths[image_index]
                    }
                })
                image_index += 1

        content_blocks.append({
            "type": "text",
            "text": prompt_parts["target"]
        })

        if image_index < len(image_paths):
            content_blocks.append({
                "type": "image",
                "source": {
                    "type": "url",
                    "url": image_paths[image_index]
                }
            })
        print("!!!!!!!!!!!!!!!!!!!!!CONTENT!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(content_blocks)

        #response = self.client.responses.create(
        #    model=self.model_name,
        #    input=[{"role": "user", "content": content_blocks}],
        #    max_output_tokens=256,
        #    temperature=0.0
        #)

        #text = response.output_text.strip()

        #if "Answer:" in text:
        #    return text.split("Answer:")[-1].strip()

        #return text.strip()"""
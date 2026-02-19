import os
from openai import OpenAI
from .base import BaseLLM


class OpenAILLM(BaseLLM):
    def __init__(self, model_name, vision):
        super().__init__(model_name, vision)
        self.client = None
        self.loaded = False

    def load(self):
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.loaded = True
        print(f"OpenAI model '{self.model_name}' loaded.")

    def generate(self, prompt_parts, max_output_tokens=256, temperature=0.0):

        if not self.loaded:
            raise RuntimeError("Model not loaded. Call `load()` first.")

        if not isinstance(prompt_parts, tuple) or len(prompt_parts) != 2:
            raise ValueError("prompt_parts must be a tuple: (instruction, blocks)")

        instruction, blocks = prompt_parts

        content_blocks = []

        for part in blocks:
            if part["type"] == "text":
                content_blocks.append({
                    "type": "input_text",
                    "text": part["text"]
                })

            elif part["type"] == "image" and self.vision:
                content_blocks.append({
                    "type": "input_image",
                    "image_url": part["source"]["url"]
                })

            elif part["type"] == "image" and not self.vision:
                continue

        if not content_blocks:
            raise ValueError("No valid content blocks to send to the model.")

        #print("instruction: ", instruction)
        #print("blocks: ", content_blocks)
                #{
                #    "role": "system",
                #    "content": instruction
                #},

        response = self.client.responses.create(
            model=self.model_name,
            instructions=instruction,
            input=[
                {
                    "role": "user",
                    "content": content_blocks
                }
            ],
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )

        text = response.output_text.strip()

        if "Answer:" in text:
            return text.split("Answer:")[-1].strip()

        return text.strip()





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
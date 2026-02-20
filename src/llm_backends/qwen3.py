import torch
from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor
from .base import BaseLLM


class Qwen3VLLLM(BaseLLM):
    def __init__(self, model_name, device="cuda" if torch.cuda.is_available() else "cpu", vision=True):
        super().__init__(model_name, vision)
        self.device = device
        self.loaded = False

    def load(self):
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            self.model_name,
            device_map="auto" if self.device.startswith("cuda") else None,
            torch_dtype=torch.bfloat16 if self.device.startswith("cuda") else torch.float32,
            trust_remote_code=True
        )

        self.model.eval()
        self.loaded = True

    def generate(self, prompt_parts, image_paths=None, max_new_tokens=256):
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call `load()` first.")

        if isinstance(prompt_parts, str):
            prompt_parts = (
                "You are a helpful assistant.",
                [{"type": "text", "text": prompt_parts}]
            )

        if not isinstance(prompt_parts, tuple) or len(prompt_parts) != 2:
            raise ValueError("prompt_parts must be tuple or string.")


        #if not self.loaded:
        #    raise RuntimeError("Model not loaded. Call `load()` first.")
        #
        #if not isinstance(prompt_parts, tuple) or len(prompt_parts) != 2:
        #    raise ValueError("prompt_parts must be a tuple: (instruction, blocks)")

        instruction, blocks = prompt_parts

        content = []

        if self.vision and image_paths:
            if not isinstance(image_paths, list):
                image_paths = [image_paths]

            for img in image_paths:
                if img:
                    content.append({
                        "type": "image",
                        "image": img
                    })

        elif isinstance(blocks, list):
            for part in blocks:
                if part["type"] == "image":
                    content.append({
                        "type": "image",
                        "image": part["source"]["url"]
                    })

        if isinstance(blocks, list):
            text_blocks = [p["text"] for p in blocks if p["type"] == "text"]
            full_text = "\n\n".join(text_blocks)
        else:
            full_text = str(blocks)

        content.append({
            "type": "text",
            "text": full_text
        })

        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": content}
            ]

        print(messages)

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True
        )

        response = output_text[0].strip()

        if "Answer:" in response:
            return response.split("Answer:")[-1].strip()

        return response


"""import torch
from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor
from .base import BaseLLM

class Qwen3VLLLM(BaseLLM):
    def __init__(self, model_name, device="cuda" if torch.cuda.is_available() else "cpu", vision=True):
        super().__init__(model_name, vision)
        self.device = device

    def load(self):
        self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            self.model_name,
            device_map="auto" if self.device.startswith("cuda") else None,
            torch_dtype=torch.bfloat16 if self.device.startswith("cuda") else torch.float32,
            trust_remote_code=True
        )
        self.model.eval()
        self.loaded = True

    def generate(self, prompt: str, image_path: str | None = None):
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call `load()` first.")

        content = []
        if self.vision and image_path:
            content.append({"type": "image", "image": image_path})
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=256)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)

        response = output_text[0].strip()
        if "Answer:" in response:
            return response.split("Answer:")[-1].strip()
        return response"""
import torch
from PIL import Image
from transformers import AutoProcessor, Llama4ForConditionalGeneration
from .base import BaseLLM


class Llama4MultimodalLLM(BaseLLM):
    def __init__(self, model_name, device="cuda" if torch.cuda.is_available() else "cpu", vision=True):
        super().__init__(model_name, vision)
        self.device = device
        self.loaded = False

    def load(self):
        self.processor = AutoProcessor.from_pretrained(self.model_name)

        self.model = Llama4ForConditionalGeneration.from_pretrained(
            self.model_name,
            attn_implementation="flex_attention",
            device_map="auto" if self.device.startswith("cuda") else None,
            torch_dtype=torch.bfloat16 if self.device.startswith("cuda") else torch.float32,
        )

        self.model.eval()
        self.loaded = True

    def generate(self, prompt_parts, image_paths=None, max_new_tokens=256, temperature=0.7, do_sample=True):
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call `load()` first.")

        if isinstance(prompt_parts, tuple) and len(prompt_parts) == 2:
            instruction, blocks = prompt_parts
            system_instruction = instruction
        else:
            blocks = prompt_parts
            system_instruction = None

        if isinstance(blocks, list):
            text_blocks = [p["text"] for p in blocks if p["type"] == "text"]
            user_text = "\n\n".join(text_blocks)
        else:
            user_text = str(blocks)

        images = []
        if image_paths:
            if not isinstance(image_paths, list):
                image_paths = [image_paths]
            for img_path in image_paths:
                images.append(Image.open(img_path))
        else:
            if isinstance(blocks, list):
                for part in blocks:
                    if part["type"] == "image":
                        if "url" in part.get("source", {}):
                            images.append(part["source"]["url"])
                        elif "path" in part.get("source", {}):
                            images.append(Image.open(part["source"]["path"]))

        content = []
        for img in images:
            content.append({"type": "image", "url": img} if isinstance(img, str) else {"type": "image", "image": img})
        content.append({"type": "text", "text": user_text})

        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": content})

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample
            )

        generated_tokens = outputs[:, inputs["input_ids"].shape[-1]:]

        response = self.processor.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )[0].strip()

        if "Answer:" in response:
            return response.split("Answer:")[-1].strip()

        return response




"""import torch
from transformers import AutoProcessor, Llama4ForConditionalGeneration
from .base import BaseLLM

class Llama4MultimodalLLM(BaseLLM):
    def __init__(self, model_name, device="cuda" if torch.cuda.is_available() else "cpu", vision=True):
        super().__init__(model_name, vision)
        self.device = device

    def load(self):
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = Llama4ForConditionalGeneration.from_pretrained(
            self.model_name,
            attn_implementation="flex_attention",
            device_map="auto" if self.device.startswith("cuda") else None,
            torch_dtype=torch.bfloat16 if self.device.startswith("cuda") else torch.float32,
        )
        self.model.eval()
        self.loaded = True

    def generate(self, prompt, image_path):
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call `load()` first.")

        content = []
        if image_path:
            content.append({"type": "image", "url": image_path})
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
            )

        response = self.processor.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:])[0]

        if "Answer:" in response:
            return response.split("Answer:")[-1].strip()
        return response.strip()"""
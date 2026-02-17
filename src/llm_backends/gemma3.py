import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from .base import BaseLLM


class Gemma3MultimodalLLM(BaseLLM):
    def __init__(self, model_name, device="cuda" if torch.cuda.is_available() else "cpu", vision=True):
        super().__init__(model_name, vision)
        self.device = device
        self.loaded = False

    def load(self):
        print(f"Loading Gemma 3 model '{self.model_name}' on {self.device}...")

        self.processor = AutoProcessor.from_pretrained(self.model_name)

        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32

        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_name,
            device_map="auto" if self.device.startswith("cuda") else None,
            torch_dtype=dtype
        )

        self.model.eval()
        self.loaded = True
        print("Model loaded successfully.")

    def generate(self, prompt_parts, image_paths=None, max_new_tokens=256, temperature=0.7, do_sample=False):
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call `load()` first.")

        if isinstance(prompt_parts, list):
            text_blocks = [p["text"] for p in prompt_parts if p["type"] == "text"]
            prompt_text = "\n\n".join(text_blocks)
        else:
            prompt_text = str(prompt_parts)

        images = []

        if image_paths:
            if not isinstance(image_paths, list):
                image_paths = [image_paths]
            images.extend(image_paths)

        elif isinstance(prompt_parts, list):
            for part in prompt_parts:
                if part["type"] == "image":
                    if "source" in part:
                        images.append(part["source"]["url"])
                    elif "image" in part:
                        images.append(part["image"])

        user_content = []

        if self.vision:
            for img in images:
                user_content.append({
                    "type": "image",
                    "image": img
                })

        user_content.append({
            "type": "text",
            "text": prompt_text
        })

        messages = []

        if hasattr(self, "system_prompt") and self.system_prompt:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}]
            })

        messages.append({
            "role": "user",
            "content": user_content
        })

        print(messages)

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(
            self.model.device,
            dtype=torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        )

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                do_sample=do_sample
            )

        generated_tokens = outputs[0][input_len:]

        decoded = self.processor.decode(
            generated_tokens,
            skip_special_tokens=True
        ).strip()

        if "Answer:" in decoded:
            return decoded.split("Answer:")[-1].strip()

        return decoded



"""import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from .base import BaseLLM


class Gemma3MultimodalLLM(BaseLLM):
    def __init__(self, model_name, device="cuda" if torch.cuda.is_available() else "cpu", vision=True):
        super().__init__(model_name, vision)
        self.device = device

    def load(self):
        print(f"Loading Gemma 3 model '{self.model_name}' on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32

        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_name,
            device_map="auto" if self.device.startswith("cuda") else None,
            torch_dtype=dtype
        )

        self.model.eval()
        self.loaded = True
        print("Model loaded successfully.")

    def generate(self, prompt, image_path):
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call `load()` first.")

        content = []

        if image_path:
            content.append({
                "type": "image",
                "image": image_path
            })

        content.append({
            "type": "text",
            "text": prompt
        })

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}]
            },
            {
                "role": "user",
                "content": content
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(
            self.model.device,
            dtype=torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        )

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False
            )

        generated_tokens = outputs[0][input_len:]

        decoded = self.processor.decode(
            generated_tokens,
            skip_special_tokens=True
        )

        if "Answer:" in decoded:
            return decoded.split("Answer:")[-1].strip()

        return decoded.strip()"""
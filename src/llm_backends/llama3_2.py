import torch
from PIL import Image
from transformers import AutoProcessor, MllamaForConditionalGeneration
from .base import BaseLLM

class Llama3_2VisionLLM(BaseLLM):
    def __init__(self, model_name, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(model_name, vision=True)
        self.device = device
        self.loaded = False

    def load(self):
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = MllamaForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16 if self.device.startswith("cuda") else torch.float32,
            device_map="auto" if self.device.startswith("cuda") else None
        )
        self.model.eval()
        self.loaded = True

    def generate(self, prompt_parts, image_paths=None, max_new_tokens=256, temperature=0.7, do_sample=True):
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call `load()` first.")

        if isinstance(prompt_parts, tuple) and len(prompt_parts) == 2:
            system_instruction, blocks = prompt_parts
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
            for path in image_paths:
                images.append(Image.open(path).convert("RGB"))
        else:
            if isinstance(blocks, list):
                for part in blocks:
                    if part["type"] == "image":
                        if "path" in part.get("source", {}):
                            images.append(Image.open(part["source"]["path"]).convert("RGB"))

        content = []
        for img in images:
            content.append({"type": "image", "image": img})
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
        response = self.processor.batch_decode(generated_tokens, skip_special_tokens=True)[0].strip()
        if "Answer:" in response:
            return response.split("Answer:")[-1].strip()
        return response
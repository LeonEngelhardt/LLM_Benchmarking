import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from .base import BaseLLM
from PIL import Image
import requests

class LlavaLLM(BaseLLM):
    def __init__(self, model_name, device="cuda" if torch.cuda.is_available() else "cpu", vision=True):
        super().__init__(model_name, vision)
        self.device = device

    def load(self):
        self.processor = LlavaNextProcessor.from_pretrained(self.model_name)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device.startswith("cuda") else torch.float32,
            low_cpu_mem_usage=True
        ).to(self.device)
        self.model.eval()
        self.loaded = True

    def generate(self, prompt: str, image_path: str | None = None):
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call `load()` first.")

        content = [{"type": "text", "text": prompt}]
        image = None
        if image_path:
            image = Image.open(requests.get(image_path, stream=True).raw).convert("RGB")
            content.append({"type": "image"})

        messages = [{"role": "user", "content": content}]
        prompt_template = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        )

        if image is not None:
            inputs = self.processor(images=image, **prompt_template).to(self.device)
        else:
            inputs = {k: v.to(self.device) for k, v in prompt_template.items()}

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=256)

        decoded = self.processor.decode(outputs[0], skip_special_tokens=True)
        if "Answer:" in decoded:
            return decoded.split("Answer:")[-1].strip()
        return decoded.strip()
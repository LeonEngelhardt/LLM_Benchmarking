import torch
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
        return response.strip()
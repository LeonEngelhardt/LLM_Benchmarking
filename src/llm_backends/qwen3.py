import torch
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
        return response
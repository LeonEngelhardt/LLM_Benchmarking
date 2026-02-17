import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from .base import BaseLLM


class LlavaOneVision7BLLM(BaseLLM):
    def __init__(self, model_name, device="cuda" if torch.cuda.is_available() else "cpu", vision=True):
        super().__init__(model_name, vision)
        self.device = device
        self.loaded = False

    def load(self):
        self.processor = AutoProcessor.from_pretrained(self.model_name)

        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16 if self.device.startswith("cuda") else torch.float32,
            low_cpu_mem_usage=True
        ).to(self.device)

        self.model.eval()
        self.loaded = True

    def generate(
        self,
        prompt_parts,
        image_paths=None,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True
    ):
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
                    images.append(part["source"]["url"])

        content = []

        for img in images:
            content.append({
                "type": "image",
                "url": img
            })

        content.append({
            "type": "text",
            "text": prompt_text
        })

        messages = [{
            "role": "user",
            "content": content
        }]

        print(messages)

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample
            )

        decoded = self.processor.decode(
            outputs[0],
            skip_special_tokens=True
        ).strip()

        if "Answer:" in decoded:
            return decoded.split("Answer:")[-1].strip()

        return decoded
import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from .base import BaseLLM
from src.utils import load_image


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

    def generate(self, prompt_parts, image_paths=None, max_new_tokens=512, temperature=0.7, do_sample=True):
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call `load()` first.")

        if isinstance(prompt_parts, tuple) and len(prompt_parts) == 2:
            instruction, blocks = prompt_parts
            system_instruction = instruction
        else:
            blocks = prompt_parts
            system_instruction = None

        content = []
        images = []

        if system_instruction:
            content.append({"type": "text", "text": system_instruction})

        if isinstance(blocks, list):
            for part in blocks:
                if not isinstance(part, dict):
                    content.append({"type": "text", "text": str(part)})
                    continue

                part_type = part.get("type")

                if part_type == "text":
                    content.append({"type": "text", "text": part.get("text", "")})
                elif part_type == "image":
                    content.append({"type": "image"})
        else:
            content.append({"type": "text", "text": str(blocks)})

        if image_paths:
            if not isinstance(image_paths, list):
                image_paths = [image_paths]
            images = [load_image(img_path) for img_path in image_paths if img_path]
        elif isinstance(blocks, list):
            for part in blocks:
                if not isinstance(part, dict) or part.get("type") != "image":
                    continue

                source = part.get("source", {})
                if "url" in source:
                    images.append(load_image(source["url"]))
                elif "path" in source:
                    images.append(load_image(source["path"]))

        messages = [{"role": "user", "content": content}]

        prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )

        inputs = self.processor(
            text=prompt,
            images=images if images else None,
            return_tensors="pt"
        ).to(
            self.device,
            torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        )

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample
            )

        generated_tokens = outputs[0][input_len:]
        decoded = self.processor.decode(generated_tokens, skip_special_tokens=True).strip()

        if "Answer:" in decoded:
            return decoded.split("Answer:")[-1].strip()

        return decoded

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
        use_cuda = torch.cuda.is_available() and self.device.startswith("cuda")
        target_device = "cuda" if use_cuda else "cpu"

        self.processor = AutoProcessor.from_pretrained(self.model_name)

        self.model = Llama4ForConditionalGeneration.from_pretrained(
            self.model_name,
            attn_implementation="flex_attention",
            device_map="auto" if use_cuda else None,
            torch_dtype=torch.bfloat16 if use_cuda else torch.float32,
        )

        if not use_cuda:
            self.model = self.model.to(target_device)

        self.model.eval()
        self.loaded = True

    def generate(self, prompt_parts, image_paths=None, max_new_tokens=512, temperature=0.7, do_sample=True):
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call `load()` first.")

        use_cuda = torch.cuda.is_available() and self.device.startswith("cuda")
        target_device = "cuda" if use_cuda else "cpu"

        if isinstance(prompt_parts, tuple) and len(prompt_parts) == 2:
            instruction, blocks = prompt_parts
            system_instruction = instruction
        else:
            blocks = prompt_parts
            system_instruction = None

        images = []
        if image_paths:
            if not isinstance(image_paths, list):
                image_paths = [image_paths]
            for img_path in image_paths:
                images.append(Image.open(img_path).convert("RGB"))
        elif isinstance(blocks, list):
            for part in blocks:
                if isinstance(part, dict) and part.get("type") == "image":
                    source = part.get("source", {})
                    if "path" in source:
                        images.append(Image.open(source["path"]).convert("RGB"))

        content = []

        if image_paths:
            content.extend([{"type": "image"} for _ in images])
        elif isinstance(blocks, list):
            for part in blocks:
                if isinstance(part, dict) and part.get("type") == "image":
                    content.append({"type": "image"})

        if isinstance(blocks, list):
            text_blocks = [p["text"] for p in blocks if isinstance(p, dict) and p.get("type") == "text"]
            user_text = "\n\n".join(text_blocks)
        else:
            user_text = str(blocks)

        content.append({"type": "text", "text": user_text})

        messages = []
        if system_instruction:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_instruction}]
            })

        messages.append({
            "role": "user",
            "content": content
        })

        prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )

        inputs = self.processor(
            text=prompt,
            images=images if images else None,
            return_tensors="pt"
        )

        inputs = {
            k: (v.to(target_device, torch.bfloat16) if use_cuda and torch.is_floating_point(v) else v.to(target_device))
            for k, v in inputs.items()
        }

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample
            )

        generated_tokens = outputs[:, input_len:]

        response = self.processor.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )[0].strip()

        if "Answer:" in response:
            return response.split("Answer:")[-1].strip()

        return response
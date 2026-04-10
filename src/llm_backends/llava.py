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
        use_cuda = torch.cuda.is_available() and self.device.startswith("cuda")
        target_device = "cuda" if use_cuda else "cpu"

        print("torch version:", torch.__version__)
        print("torch cuda runtime:", torch.version.cuda)
        print("cuda available:", torch.cuda.is_available())
        print("cuda device count:", torch.cuda.device_count())
        if torch.cuda.is_available():
            print("device 0:", torch.cuda.get_device_name(0))

        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            use_fast=True,
        )

        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.model_max_length = 2048
            print("forced tokenizer max length:", self.processor.tokenizer.model_max_length)

        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16 if use_cuda else torch.float32,
            low_cpu_mem_usage=True,
        ).to(target_device)

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

        content = []
        images = []

        if system_instruction:
            content.append({"type": "text", "text": str(system_instruction)})

        if isinstance(blocks, list):
            for part in blocks:
                if not isinstance(part, dict):
                    content.append({"type": "text", "text": str(part)})
                    continue

                part_type = part.get("type")

                if part_type == "text":
                    content.append({"type": "text", "text": str(part.get("text", ""))})
                elif part_type == "image":
                    content.append({"type": "image"})
        else:
            content.append({"type": "text", "text": str(blocks)})

        if image_paths:
            if not isinstance(image_paths, list):
                image_paths = [image_paths]

            loaded_images = [load_image(img_path) for img_path in image_paths if img_path]
            images.extend(loaded_images)

            content = [{"type": "image"} for _ in loaded_images] + [
                item for item in content if item.get("type") == "text"
            ]

        elif isinstance(blocks, list):
            for part in blocks:
                if not isinstance(part, dict) or part.get("type") != "image":
                    continue

                source = part.get("source", {})
                if isinstance(source, dict):
                    if "url" in source:
                        images.append(load_image(source["url"]))
                    elif "path" in source:
                        images.append(load_image(source["path"]))
                elif isinstance(source, str):
                    images.append(load_image(source))

        messages = [{"role": "user", "content": content}]

        prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        # Truncate the prompt text BEFORE the processor builds final tensors
        if hasattr(self.processor, "tokenizer"):
            tokenized_prompt = self.processor.tokenizer(
                prompt,
                truncation=True,
                max_length=2048,
                return_tensors="pt",
            )
            prompt = self.processor.tokenizer.decode(
                tokenized_prompt["input_ids"][0],
                skip_special_tokens=False,
            )

        inputs = self.processor(
            text=prompt,
            images=images if images else None,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )

        inputs = {
            k: (
                v.to(target_device, dtype=torch.bfloat16)
                if use_cuda and torch.is_tensor(v) and torch.is_floating_point(v)
                else v.to(target_device)
            )
            for k, v in inputs.items()
        }

        input_len = inputs["input_ids"].shape[-1]
        print("tokenizer max length:", getattr(self.processor.tokenizer, "model_max_length", "unknown"))
        print("actual input length:", input_len)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.processor.tokenizer.eos_token_id if hasattr(self.processor, "tokenizer") else None,
            )

        generated_tokens = outputs[:, input_len:]

        decoded = self.processor.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
        )[0].strip()

        if "Answer:" in decoded:
            return decoded.split("Answer:")[-1].strip()

        return decoded
import torch
from PIL import Image
from transformers import AutoProcessor, Llama4ForConditionalGeneration
from .base import BaseLLM


class Llama4MultimodalLLM(BaseLLM):
    def __init__(self, model_name, device="cuda", vision=True):
        super().__init__(model_name, vision)
        self.device = device
        self.loaded = False

    def load(self):
        print("torch version:", torch.__version__)
        print("torch cuda runtime:", torch.version.cuda)
        print("cuda available:", torch.cuda.is_available())
        print("cuda device count:", torch.cuda.device_count())
        print("requested device:", self.device)

        if not (torch.cuda.is_available() and self.device.startswith("cuda")):
            raise RuntimeError(
                "CUDA is not available for Llama-4 on this job; refusing CPU fallback to avoid OOM."
            )

        print("device 0:", torch.cuda.get_device_name(0))

        self.processor = AutoProcessor.from_pretrained(self.model_name)

        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.model_max_length = 2048
            print("forced tokenizer max length:", self.processor.tokenizer.model_max_length)

        max_memory = None
        if self.device.startswith("cuda"):
            gpu_count = torch.cuda.device_count()
            max_memory = {"cpu": "160GiB"}
            for gpu_idx in range(gpu_count):
                total_bytes = torch.cuda.get_device_properties(gpu_idx).total_memory
                total_gib = total_bytes / (1024 ** 3)
                # Keep a few GiB free per GPU for activations, temporary buffers, and CUDA runtime.
                usable_gib = max(1, int(total_gib - 4))
                max_memory[gpu_idx] = f"{usable_gib}GiB"
                print(f"device {gpu_idx} total memory (GiB):", round(total_gib, 2))
                print(f"device {gpu_idx} usable memory cap (GiB):", usable_gib)

        self.model = Llama4ForConditionalGeneration.from_pretrained(
            self.model_name,
            device_map="auto",
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            max_memory=max_memory,
            offload_folder="results/llama4_offload",
        )

        self.model.eval()
        self.loaded = True

    def generate(
        self,
        prompt_parts,
        image_paths=None,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
    ):
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call `load()` first.")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA became unavailable after model load.")

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
                    if isinstance(source, dict) and "path" in source:
                        images.append(Image.open(source["path"]).convert("RGB"))

        content = []

        if image_paths:
            content.extend([{"type": "image"} for _ in images])
        elif isinstance(blocks, list):
            for part in blocks:
                if isinstance(part, dict) and part.get("type") == "image":
                    content.append({"type": "image"})

        if isinstance(blocks, list):
            text_blocks = [
                p["text"]
                for p in blocks
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            user_text = "\n\n".join(text_blocks)
        else:
            user_text = str(blocks)

        content.append({"type": "text", "text": user_text})

        messages = []
        if system_instruction:
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": str(system_instruction)}],
                }
            )

        messages.append(
            {
                "role": "user",
                "content": content,
            }
        )

        prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

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

        first_device = next(self.model.parameters()).device
        inputs = {
            k: (
                v.to(first_device, dtype=torch.bfloat16)
                if torch.is_tensor(v) and torch.is_floating_point(v)
                else v.to(first_device)
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
            )

        generated_tokens = outputs[:, input_len:]

        response = self.processor.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
        )[0].strip()

        if "Answer:" in response:
            return response.split("Answer:")[-1].strip()

        return response

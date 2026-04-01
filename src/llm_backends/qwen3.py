import os
import torch
from transformers import AutoProcessor
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base import BaseLLM


class Qwen3VLLLM(BaseLLM):
    def __init__(self, model_name, device="cuda" if torch.cuda.is_available() else "cpu", vision=True):
        super().__init__(model_name, vision)
        self.device = device
        self.loaded = False

        #self.cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        self.cache_dir = os.path.join(
            os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
            "hub"
        )
    
    def load(self):
        if self.vision:
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                cache_dir=self.cache_dir
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                cache_dir=self.cache_dir
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto" if self.device.startswith("cuda") else None,
            torch_dtype=torch.bfloat16 if self.device.startswith("cuda") else torch.float32,
            trust_remote_code=True,
            cache_dir=self.cache_dir
        )

        self.model.eval()
        self.loaded = True
    

    def generate(self, prompt_parts, image_paths=None, max_new_tokens=256):
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call `load()` first.")

        direct_text_prompt = None
        if isinstance(prompt_parts, str):
            if not self.vision and self.model_name == "Qwen/Qwen3-4B-Instruct-2507":
                direct_text_prompt = prompt_parts.strip()
            else:
                prompt_parts = (
                    "You are a helpful assistant.",
                    [{"type": "text", "text": prompt_parts}]
                )

        if direct_text_prompt is None and (not isinstance(prompt_parts, tuple) or len(prompt_parts) != 2):
            raise ValueError("prompt_parts must be tuple or string.")


        #if not self.loaded:
        #    raise RuntimeError("Model not loaded. Call `load()` first.")
        #
        #if not isinstance(prompt_parts, tuple) or len(prompt_parts) != 2:
        #    raise ValueError("prompt_parts must be a tuple: (instruction, blocks)")

        if not self.vision:
            if self.model_name == "Qwen/Qwen3-4B-Instruct-2507":
                if direct_text_prompt is not None:
                    judge_prompt = direct_text_prompt
                else:
                    instruction, blocks = prompt_parts
                    if isinstance(blocks, list):
                        text_blocks = [p["text"] for p in blocks if p["type"] == "text"]
                        full_text = "\n\n".join(text_blocks)
                    else:
                        full_text = str(blocks)
                    judge_prompt = f"{instruction}\n\n{full_text}".strip()

                print(f"[DEBUG][Qwen3Judge] prompt_preview={judge_prompt[:400]!r}")
                messages = [
                    {"role": "user", "content": judge_prompt}
                ]
            else:
                instruction, blocks = prompt_parts
                if isinstance(blocks, list):
                    text_blocks = [p["text"] for p in blocks if p["type"] == "text"]
                    full_text = "\n\n".join(text_blocks)
                else:
                    full_text = str(blocks)
                messages = [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": full_text}
                ]

            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        else:
            instruction, blocks = prompt_parts
            if isinstance(blocks, list):
                text_blocks = [p["text"] for p in blocks if p["type"] == "text"]
                full_text = "\n\n".join(text_blocks)
            else:
                full_text = str(blocks)
            content = []

            if image_paths:
                if not isinstance(image_paths, list):
                    image_paths = [image_paths]

                for img in image_paths:
                    if img:
                        content.append({
                            "type": "image",
                            "image": img
                        })

            elif isinstance(blocks, list):
                for part in blocks:
                    if part["type"] == "image":
                        content.append({
                            "type": "image",
                            "image": part["source"]["url"]
                        })

            content.append({
                "type": "text",
                "text": full_text
            })

            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": content}
            ]

            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=False
            ).to(self.model.device)

        #with torch.no_grad():
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        if self.vision:
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True
            )
        else:
            output_text = self.tokenizer.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True
            )

        response = output_text[0].strip()

        if "Answer:" in response:
            return response.split("Answer:")[-1].strip()

        return response

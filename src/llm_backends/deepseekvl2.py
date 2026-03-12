import os
import torch
from transformers import AutoModelForCausalLM
from .base import BaseLLM
import sys
import subprocess
from PIL import Image
import requests
from io import BytesIO


class DeepSeekVLV2LLM(BaseLLM):
    def __init__(self, model_name, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(model_name, vision=True)
        self.device = device
        self.loaded = False
        self.processor = None
        self.tokenizer = None
        self.model = None
        
        #self.cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        self.cache_dir = os.path.join(
            os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
            "hub"
        )

    def ensure_deepseek_vl2_installed(self):
        try:
            import deepseek_vl2
        except ModuleNotFoundError:
            import sys, subprocess
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-e",
                "git+https://github.com/deepseek-ai/deepseek-vl2.git#egg=deepseek-vl2"
            ])
        finally:
            global DeepseekVLV2Processor
            from deepseek_vl2.models import DeepseekVLV2Processor

    def load(self):
        #self.ensure_deepseek_vl2_installed()
        from deepseek_vl2.models import DeepseekVLV2Processor

        self.processor = DeepseekVLV2Processor.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir
        )
        self.tokenizer = self.processor.tokenizer

        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            cache_dir=self.cache_dir
        ).to(dtype).to(self.device).eval()

        self.loaded = True


    def _load_images(self, image_paths):
        pil_images = []
        for path in image_paths:
            if path.startswith("http"):
                response = requests.get(path, timeout=10)
                img = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                img = Image.open(path).convert("RGB")
            pil_images.append(img)
        return pil_images
    
    def generate(self, prompt_parts, image_paths=None,
                max_new_tokens=512,
                temperature=0.7):

        if not self.loaded:
            raise RuntimeError("Model not loaded. Call `load()` first.")

        if isinstance(prompt_parts, tuple) and len(prompt_parts) == 2:
            instruction, blocks = prompt_parts
            system_prompt = instruction
        else:
            system_prompt = ""
            blocks = prompt_parts

        if isinstance(blocks, list):
            text_blocks = [p["text"] for p in blocks if p.get("type") == "text"]
            user_content = "\n\n".join(text_blocks)
        else:
            user_content = str(blocks)

        if image_paths:
            user_content = "<image>\n" + user_content

        conversation = [
            {
                "role": "<|User|>",
                "content": user_content,
                "images": image_paths or []
            },
            {
                "role": "<|Assistant|>",
                "content": ""
            }
        ]


        pil_images = self._load_images(image_paths) if image_paths else []

        prepare_inputs = self.processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt=system_prompt
        ).to(self.device)

        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

        with torch.inference_mode():
            outputs = self.model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                use_cache=True
            )

        output_ids = outputs[0]
        input_length = prepare_inputs.input_ids.shape[1]
        generated_ids = output_ids[input_length:]

        response = self.tokenizer.decode(
            generated_ids.cpu().tolist(),
            skip_special_tokens=True
        ).strip()

        return response
    





    """def generate(self, prompt_parts, image_paths=None,
                 max_new_tokens=512,
                 temperature=0.7):

        if not self.loaded:
            raise RuntimeError("Model not loaded. Call `load()` first.")

        if isinstance(prompt_parts, tuple) and len(prompt_parts) == 2:
            instruction, blocks = prompt_parts
            system_message = {"role": "system", "content": instruction}
        else:
            system_message = None
            blocks = prompt_parts

        if isinstance(blocks, list):
            text_blocks = [p["text"] for p in blocks if p.get("type") == "text"]
            user_content = "\n\n".join(text_blocks)
        else:
            user_content = str(blocks)

        if image_paths:
            user_content = "<image>\n" + user_content

        conversation = [
            {
                "role": "<|User|>",
                "content": user_content,
                "images": image_paths or [],
            }
        ]

        pil_images = self._load_images(image_paths) if image_paths else []

        prepare_inputs = self.processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt=system_message["content"] if system_message else ""
        ).to(self.device)

        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

        with torch.inference_mode():
            outputs = self.model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                use_cache=True
            )

        output_ids = outputs[0]
        input_length = prepare_inputs.input_ids.shape[1]
        generated_ids = output_ids[input_length:]

        response = self.tokenizer.decode(
            generated_ids.cpu().tolist(),
            skip_special_tokens=True
        ).strip()

        return response"""
    
    """def ensure_deepseek_vl2_installed(self):
        try:
            import deepseek_vl2
        except ModuleNotFoundError:
            print("DeepSeek-VL2 not found --> installing from GitHub-Repo...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-e",
                "git+https://github.com/deepseek-ai/deepseek-vl2.git#egg=deepseek-vl2"
            ])
        finally:
            global DeepseekVLV2Processor
            from deepseek_vl2.models import DeepseekVLV2Processor

    def load(self):
        self.ensure_deepseek_vl2_installed()

        from deepseek_vl2.models import DeepseekVLV2Processor

        self.processor = DeepseekVLV2Processor.from_pretrained(self.model_name)
        self.tokenizer = self.processor.tokenizer

        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True
        ).to(dtype).to(self.device).eval()

        self.loaded = True"""
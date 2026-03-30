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
        self.device = "cuda"
        self.loaded = False
        self.processor = None
        self.tokenizer = None
        self.model = None
        self.model_dtype = torch.bfloat16
        
        #self.cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        self.cache_dir = os.path.join(
            os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
            "hub"
        )

    def _cast_model_to_runtime_dtype(self):
        self.model = self.model.to(device=self.device, dtype=self.model_dtype)

        for submodule_name in ("vision", "projector", "language"):
            submodule = getattr(self.model, submodule_name, None)
            if submodule is not None:
                submodule.to(device=self.device, dtype=self.model_dtype)

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

        self.model_dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=self.model_dtype,
            cache_dir=self.cache_dir
        )
        self._cast_model_to_runtime_dtype()
        self.model.eval()

        self.loaded = True


    def _load_images(self, image_paths):
        pil_images = []
        for path in image_paths:
            if path.startswith("http"):
                response = requests.get(path, timeout=10)
                response.raise_for_status()
                try:
                    img = Image.open(BytesIO(response.content)).convert("RGB")
                except Exception as exc:
                    raise ValueError(f"Failed to decode remote image: {path}") from exc
            else:
                try:
                    img = Image.open(path).convert("RGB")
                except Exception as exc:
                    raise ValueError(f"Failed to decode local image: {path}") from exc
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
        )

        for key, value in vars(prepare_inputs).items():
            if torch.is_tensor(value):
                value = value.to(self.device)
                if torch.is_floating_point(value):
                    value = value.to(self.model_dtype)
                setattr(prepare_inputs, key, value)

        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

        with torch.no_grad():
        #with torch.inference_mode(): --> deepseek_vl2 needs no_grad according to cluster-log!
            outputs = self.model.generate(
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

        trimmed_response = self.tokenizer.decode(
            generated_ids.cpu().tolist(),
            skip_special_tokens=True
        ).strip()

        full_response = self.tokenizer.decode(
            output_ids.cpu().tolist(),
            skip_special_tokens=True
        ).strip()

        print(
            f"[DEBUG][DeepSeekVL2] input_len={input_length} "
            f"output_len={output_ids.shape[0]} generated_len={generated_ids.shape[0]}"
        )
        print(f"[DEBUG][DeepSeekVL2] trimmed_response_preview={trimmed_response[:200]!r}")
        print(f"[DEBUG][DeepSeekVL2] full_response_preview={full_response[:200]!r}")

        if trimmed_response:
            return trimmed_response

        if full_response:
            return full_response

        return ""

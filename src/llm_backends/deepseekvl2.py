import torch
from transformers import AutoModelForCausalLM
from .base import BaseLLM
import sys
import subprocess
import deepseek_vl2
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images

class DeepSeekVLV2LLM(BaseLLM):
    def __init__(self, model_name, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(model_name, vision=True)
        self.device = device
        self.loaded = False
        self.processor = None
        self.tokenizer = None
        self.model = None

    def ensure_deepseek_vl2_installed(self):
        try:
            import deepseek_vl2
        except ModuleNotFoundError:
            print("DeepSeek-VL2 not found --> installing from GitHub-Repo...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-e",
                "git+https://github.com/deepseek-ai/deepseek-vl2.git#egg=deepseek-vl2"
            ])
        finally:
            global DeepseekVLV2Processor, DeepseekVLV2ForCausalLM, load_pil_images
            from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
            from deepseek_vl2.utils.io import load_pil_images

    def load(self):
        self.ensure_deepseek_vl2_installed()

        self.processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(
            self.model_name
        )
        self.tokenizer = self.processor.tokenizer

        self.model: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True
        ).to(torch.bfloat16).to(self.device).eval()

        self.loaded = True

    def generate(self, prompt_parts, image_paths=None, max_new_tokens=512, temperature=0.7):
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

        conversation = [
            {"role": "<|User|>", "content": user_content, "images": image_paths or []},
        ]

        print(conversation)

        pil_images = load_pil_images(conversation) if image_paths else []
        prepare_inputs = self.processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt=system_message["content"] if system_message else ""
        ).to(self.device)

        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

        with torch.no_grad():
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

        response = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        return response
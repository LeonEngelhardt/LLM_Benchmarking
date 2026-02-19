import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .base import BaseLLM


class MistralLLM(BaseLLM):
    def __init__(self, model_name, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(model_name, vision=False)
        self.device = device
        self.loaded = False

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16 if "cuda" in self.device else torch.float32,
            device_map="auto" if "cuda" in self.device else None,
            trust_remote_code=True
        )

        self.model.eval()
        self.loaded = True

    def generate(self, prompt_parts, image_paths=None, max_new_tokens=256, temperature=0.7):

        if not self.loaded:
            raise RuntimeError("Model not loaded. Call `load()` first.")

        if isinstance(prompt_parts, tuple) and len(prompt_parts) == 2:
            instruction, blocks = prompt_parts
            system_message = {"role": "system", "content": instruction}
        else:
            system_message = None
            blocks = prompt_parts

        if isinstance(blocks, list):
            text_blocks = [p["text"] for p in blocks if p["type"] == "text"]
            user_content = "\n\n".join(text_blocks)
        else:
            user_content = str(blocks)

        user_message = {"role": "user", "content": user_content}

        messages = [m for m in (system_message, user_message) if m is not None]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        input_length = inputs["input_ids"].shape[-1]
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        ).strip()

        return response
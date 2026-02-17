import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .base import BaseLLM


class HFTextLLM(BaseLLM):
    def __init__(self, model_name, device="cpu", hf_token=None):
        super().__init__(model_name, vision=False)
        self.device = device
        self.hf_token = hf_token
        self.loaded = False

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_auth_token=self.hf_token
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            use_auth_token=self.hf_token
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

        self.model.to(self.device)
        self.model.eval()
        self.loaded = True

    def generate(self, prompt_parts, image_paths=None, max_new_tokens=256, temperature=0.7):

        if not self.loaded:
            raise RuntimeError("Model not loaded. Call `load()` first.")

        if isinstance(prompt_parts, list):
            text_blocks = [p["text"] for p in prompt_parts if p["type"] == "text"]
            prompt_text = "\n\n".join(text_blocks)
        else:
            prompt_text = str(prompt_parts)

        if "llama" in self.model_name.lower():
            prompt_text = f"<s>[INST] {prompt_text} [/INST]"

        print("Prompt: ", prompt_text)

        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id
            )

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text.strip()
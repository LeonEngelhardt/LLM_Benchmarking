import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .base import BaseLLM

class HFTextLLM(BaseLLM):
    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=self.hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            token=self.hf_token,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        ).to(self.device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt: str, image_path=None):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=0.0,
            pad_token_id=self.tokenizer.eos_token_id
        )

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text[len(prompt):].strip()

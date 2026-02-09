import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .base import BaseLLM

class HFTextLLM(BaseLLM):
    def __init__(self, model_name, device, hf_token):
        super().__init__(model_name, vision=False)
        self.device = device
        self.hf_token = hf_token

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=self.hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, token=self.hf_token)
        self.model.to(self.device)
        self.loaded = True
    
    def generate(self, prompt, image_path=None):
        if "llama-2" in self.model_name.lower():
            prompt = f"<s>[INST] {prompt} [/INST]"

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


    """def generate(self, prompt, image_path=None):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=128)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)"""
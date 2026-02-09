import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .base import BaseLLM

class HFTextLLM(BaseLLM):
    def __init__(self, model_name, device, hf_token=None): # hf_token
        super().__init__(model_name, vision=False)
        self.device = device
        self.hf_token = hf_token

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=self.hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, token=self.hf_token)

        # GPT2 & similar models --> only for local testing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

        self.model.to(self.device)
        self.loaded = True
    
    def generate(self, prompt, image_path=None):
        if "llama-2" in self.model_name.lower():
            prompt = f"<s>[INST] {prompt} [/INST]"

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.device)

        # to save ressources --> might not be necessary! --> if not reimplement outputs... below
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True
            )

        """
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7
        )"""

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
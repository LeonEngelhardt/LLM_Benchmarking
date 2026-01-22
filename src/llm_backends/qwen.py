from .openrouter import OpenRouterBackend

class QwenBackend(OpenRouterBackend):
    pass

"""from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class QwenLLM:
    def __init__(self, model_name="Qwen/Qwen3-235B-A22B-Instruct-2507", device="cuda", hf_token=None):
        self.model_name = model_name
        self.device = device
        self.hf_token = hf_token

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=self.hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            token=self.hf_token
        )

    def generate(self, prompt: str, **kwargs) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False
        )

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text[len(prompt):].strip()"""
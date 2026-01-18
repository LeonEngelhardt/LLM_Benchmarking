from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class BaseLLM:
    def __init__(self, model_name: str, device="cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None

    def load_model(self):
        raise NotImplementedError

    def generate(self, prompt: str, max_length=200):
        raise NotImplementedError

class HuggingFaceLLM(BaseLLM):
    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map="auto", torch_dtype=torch.float16
        )

    def generate(self, prompt: str, max_length=200):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_length)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()

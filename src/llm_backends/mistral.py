import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class MistralLLM:
    def __init__(self, model_name, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.loaded = False

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16 if "cuda" in self.device else torch.float32,
            device_map="auto" if "cuda" in self.device else None
        )
        self.model.eval()
        self.loaded = True

    def generate(self, prompt):
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call `load()` first.")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
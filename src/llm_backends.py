import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BlipProcessor, BlipForConditionalGeneration
)
from PIL import Image


class BaseLLM:
    def generate(self, prompt, image_path=None):
        raise NotImplementedError


class TextLLM(BaseLLM):
    def __init__(self, model_name, device, hf_token=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token).to(device)
        self.device = device

    def generate(self, prompt, image_path=None):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=50)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class BlipVLM(BaseLLM):
    def __init__(self, model_name, device):
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
        self.device = device

    def generate(self, prompt, image_path=None):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=50)
        return self.processor.decode(out[0], skip_special_tokens=True)

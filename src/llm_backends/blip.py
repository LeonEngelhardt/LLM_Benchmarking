from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from .base import BaseLLM

class BlipLLM(BaseLLM):
    def __init__(self, model_name, device):
        super().__init__(model_name, vision=True)
        self.device = device

    def load(self):
        self.processor = BlipProcessor.from_pretrained(self.model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.loaded = True

    def generate(self, prompt, image_path=None):
        if not image_path:
            return self.model.generate(prompt)

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            return self.model.generate(prompt)

        inputs = self.processor(image, prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs)
        return self.processor.decode(output[0], skip_special_tokens=True)


"""from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from .base import BaseLLM

class BlipLLM(BaseLLM):
    def __init__(self, model_name, device):
        super().__init__(model_name, vision=True)
        self.device = device

    def load(self):
        self.processor = BlipProcessor.from_pretrained(self.model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.loaded = True

    def generate(self, prompt, image_path=None):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(image, prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs)
        return self.processor.decode(output[0], skip_special_tokens=True)"""

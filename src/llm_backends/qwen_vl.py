from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
from .base import BaseLLM

class QwenVLLM(BaseLLM):
    def __init__(self, model_name, device):
        super().__init__(model_name, vision=True)
        self.device = device

    def load(self):
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(self.model_name).to(self.device)
        self.loaded = True

    def generate(self, prompt, image_path=None):
        if not image_path:
            return self.model.generate(prompt)

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            return self.model.generate(prompt)
        
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=128)

        return self.processor.decode(output[0], skip_special_tokens=True)
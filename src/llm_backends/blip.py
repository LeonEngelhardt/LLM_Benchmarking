# local --> can be deleted
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from .base import BaseLLM

class BlipLLM(BaseLLM):
    def load_model(self):
        self.processor = BlipProcessor.from_pretrained(self.model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)

    def generate(self, prompt: str, image_path=None):
        if image_path is None:
            raise ValueError("BLIP requires an image_path.")

        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=50
        )

        return self.processor.decode(outputs[0], skip_special_tokens=True)

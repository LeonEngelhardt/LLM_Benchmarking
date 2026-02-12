from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from .base import BaseLLM
import torch

class BlipLLM(BaseLLM):
    def __init__(self, model_name, device="cpu"):
        super().__init__(model_name, vision=True)
        self.device = device
        self.loaded = False

    def load(self):
        self.processor = BlipProcessor.from_pretrained(self.model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        self.loaded = True
        self._dummy_image = Image.new("RGB", (224, 224), color="white")

    def generate(self, prompt, image_path=None, max_new_tokens=512, temperature=0.7):
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call `load()` first.")

        if image_path is None:
            images = [self._dummy_image]
        elif isinstance(image_path, list):
            images = []
            for path in image_path:
                try:
                    img = Image.open(path).convert("RGB")
                    images.append(img)
                except Exception as e:
                    print(f"[WARN] Could not open image '{path}', using dummy: {e}")
                    images.append(self._dummy_image)
        else:
            try:
                images = [Image.open(image_path).convert("RGB")]
            except Exception as e:
                print(f"[WARN] Could not open image '{image_path}', using dummy: {e}")
                images = [self._dummy_image]

        inputs = self.processor(
            images=images,
            text=[prompt] * len(images),
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature
            )

        texts = [self.processor.decode(out, skip_special_tokens=True).strip() for out in outputs]

        final_text = " ".join(texts) if len(texts) > 1 else texts[0]

        return final_text




"""from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from .base import BaseLLM
import torch

class BlipLLM(BaseLLM):
    def __init__(self, model_name, device="cpu"):
        super().__init__(model_name, vision=True)
        self.device = device
        self.loaded = False

    def load(self):
        self.processor = BlipProcessor.from_pretrained(self.model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        self.loaded = True

    def generate(self, prompt, image_path=None, max_new_tokens=512, temperature=0.7):
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call `load()` first.")

        if not image_path:
            
            raise ValueError("BLIP requires an image_path input for generation.")

        if isinstance(image_path, list):
            images = []
            for path in image_path:
                try:
                    img = Image.open(path).convert("RGB")
                    images.append(img)
                except Exception as e:
                    raise RuntimeError(f"Could not open image '{path}': {e}")
        else:
            try:
                images = [Image.open(image_path).convert("RGB")]
            except Exception as e:
                raise RuntimeError(f"Could not open image '{image_path}': {e}")

        inputs = self.processor(
            images=images,
            text=[prompt] * len(images),
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature
            )

        texts = [self.processor.decode(out, skip_special_tokens=True).strip() for out in outputs]
        return " ".join(texts)"""
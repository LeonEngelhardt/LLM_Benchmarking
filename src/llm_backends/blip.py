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

    def generate(self, prompt_parts, image_paths=None, max_new_tokens=512, temperature=0.7):
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call `load()` first.")

        # ------------------------
        # TEXT extrahieren
        # ------------------------
        if isinstance(prompt_parts, list):
            text_blocks = [p["text"] for p in prompt_parts if p["type"] == "text"]
            prompt_text = "\n\n".join(text_blocks)
        else:
            prompt_text = str(prompt_parts)

        # ------------------------
        # IMAGE bestimmen
        # ------------------------
        image_path = None

        if image_paths:
            image_path = image_paths[-1]
        else:
            if isinstance(prompt_parts, list):
                image_blocks = [p for p in prompt_parts if p["type"] == "image"]
                if image_blocks:
                    image_path = image_blocks[-1]["source"]["url"]

        # ------------------------
        # Dummy-Bild erzeugen, falls kein Bild vorhanden
        # ------------------------
        if not image_path:
            print("[BLIP] Kein Bild gefunden – benutze Dummy-Bild.")
            image = Image.new("RGB", (224, 224), color=(128, 128, 128))  # graues Dummy-Bild
        else:
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                raise RuntimeError(f"Could not open image '{image_path}': {e}")

        # ------------------------
        # Debug-Ausgabe
        # ------------------------
        print("Image: ", image)
        print("Prompt: ", prompt_text)

        # ------------------------
        # Eingaben für BLIP
        # ------------------------
        inputs = self.processor(
            images=image,
            text=prompt_text,
            return_tensors="pt"
        ).to(self.device)

        # ------------------------
        # Generation
        # ------------------------
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature
            )

        text = self.processor.decode(outputs[0], skip_special_tokens=True)
        return text.strip()
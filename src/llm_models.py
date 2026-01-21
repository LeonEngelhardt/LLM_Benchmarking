import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image


class HuggingFaceLLM:
    def __init__(self, model_name, hf_token=None, device="cpu"):
        self.model_name = model_name
        self.hf_token = hf_token
        self.device = device
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.is_vision = False

    def load_model(self):
        if "blip" in self.model_name.lower():
            self.is_vision = True
            self.processor = BlipProcessor.from_pretrained(self.model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
            print("[INFO] Vision-Text Modell (BLIP) geladen.")
        else:
            self.is_vision = False
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=self.hf_token)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=self.hf_token,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            print("[INFO] Text LLM geladen.")

    def generate(self, prompt, image_path=None, max_length=128):
        if self.is_vision:
            if image_path is None:
                raise ValueError("BLIP benötigt immer ein Bild (image_path darf nicht None sein).")

            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    pixel_values=inputs["pixel_values"],
                    input_ids=inputs.get("input_ids"),
                    max_new_tokens=max_length
                )

            return self.processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, max_new_tokens=max_length)
            return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
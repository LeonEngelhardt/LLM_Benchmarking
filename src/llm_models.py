from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from PIL import Image
import requests
from io import BytesIO
import torch

class HuggingFaceMultimodalLLM:
    def __init__(self, model_name: str, device="cpu"):
        self.model_name = model_name
        #self.device = device
        #self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"
        self.model = None
        self.tokenizer = None
        self.processor = None

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )

    def generate(self, prompt: str, image_path: str = None, max_length=200):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        if image_path:
            # load picture (local or URL)
            if image_path.startswith("http"):
                response = requests.get(image_path)
                image = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                image = Image.open(image_path).convert("RGB")
            
            image_inputs = self.processor(images=image, return_tensors="pt").to(self.model.device)
            inputs.update(image_inputs)

        outputs = self.model.generate(**inputs, max_new_tokens=max_length)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()

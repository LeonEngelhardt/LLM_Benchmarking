import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from .base import BaseLLM


class Gemma3MultimodalLLM(BaseLLM):
    def __init__(self, model_name, device="cuda" if torch.cuda.is_available() else "cpu", vision=True):
        super().__init__(model_name, vision)
        self.device = device

    def load(self):
        print(f"Loading Gemma 3 model '{self.model_name}' on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32

        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_name,
            device_map="auto" if self.device.startswith("cuda") else None,
            torch_dtype=dtype
        )

        self.model.eval()
        self.loaded = True
        print("Model loaded successfully.")

    def generate(self, prompt, image_path):
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call `load()` first.")

        content = []

        if image_path:
            content.append({
                "type": "image",
                "image": image_path
            })

        content.append({
            "type": "text",
            "text": prompt
        })

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}]
            },
            {
                "role": "user",
                "content": content
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(
            self.model.device,
            dtype=torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        )

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False
            )

        generated_tokens = outputs[0][input_len:]

        decoded = self.processor.decode(
            generated_tokens,
            skip_special_tokens=True
        )

        if "Answer:" in decoded:
            return decoded.split("Answer:")[-1].strip()

        return decoded.strip()



"""google/gemma-3n-e2b-it
import torch
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
from .base import BaseLLM

class Gemma3nMultimodalLLM(BaseLLM):
    def __init__(self, model_name, device="cuda" if torch.cuda.is_available() else "cpu", vision=True):
        super().__init__(model_name, vision)
        self.device = device
        self.loaded = False

    def load(self):
        print(f"Loading Gemma 3n model '{self.model_name}' on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.model = Gemma3nForConditionalGeneration.from_pretrained(
            self.model_name,
            device_map="auto" if self.device.startswith("cuda") else None,
            torch_dtype=dtype
        )
        self.model.eval()
        self.loaded = True
        print("Model loaded successfully.")

    def generate(self, prompt, image_path=None, max_new_tokens=256):
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call `load()` first.")

        content = []

        if image_path:
            content.append({"type": "image", "url": image_path})

        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16 if self.device.startswith("cuda") else torch.float32)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )

        generated = self.processor.batch_decode(
            outputs[:, inputs["input_ids"].shape[-1]:]
        )[0]

        if "Answer:" in generated:
            return generated.split("Answer:")[-1].strip()
        return generated.strip()"""
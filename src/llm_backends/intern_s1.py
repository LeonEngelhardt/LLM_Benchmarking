import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from .base import BaseLLM


class InternS1LLM(BaseLLM):
    def __init__(self, model_name, device="cuda" if torch.cuda.is_available() else "cpu", vision=True):
        super().__init__(model_name, vision)
        self.device = device
        self.loaded = False

    def load(self):
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto" if self.device.startswith("cuda") else None,
            torch_dtype=torch.bfloat16 if self.device.startswith("cuda") else torch.float32,
            trust_remote_code=True
        )

        self.model.eval()
        self.loaded = True

    def generate(self, prompt_parts, image_paths=None, max_new_tokens=256, temperature=0.7, top_p=1.0, top_k=50, do_sample=True):
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call `load()` first.")
        
        if isinstance(prompt_parts, tuple) and len(prompt_parts) == 2:
            system_instruction, blocks = prompt_parts
        else:
            blocks = prompt_parts
            system_instruction = None

        if isinstance(blocks, list):
            text_blocks = [p["text"] for p in blocks if p["type"] == "text"]
            prompt_text = "\n\n".join(text_blocks)
        else:
            prompt_text = str(blocks)

        images = []
        if image_paths:
            if not isinstance(image_paths, list):
                image_paths = [image_paths]
            images.extend(image_paths)
        elif isinstance(blocks, list):
            for part in blocks:
                if part["type"] == "image":
                    if "url" in part.get("source", {}):
                        images.append(part["source"]["url"])
                    elif "path" in part.get("source", {}):
                        images.append(part["source"]["path"])

        content = []
        if self.vision:
            for img in images:
                content.append({"type": "image", "url": img})

        content.append({"type": "text", "text": prompt_text})

        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": content})

        print(messages)

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k
            )

        generated_tokens = outputs[:, inputs["input_ids"].shape[-1]:]

        response = self.processor.decode(
            generated_tokens[0],
            skip_special_tokens=True
        ).strip()

        if "Answer:" in response:
            return response.split("Answer:")[-1].strip()

        return response




"""import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from .base import BaseLLM

class InternS1LLM(BaseLLM):
    def __init__(self, model_name, device="cuda" if torch.cuda.is_available() else "cpu", vision=True):
        super().__init__(model_name, vision)
        self.device = device

    def load(self):
        self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto" if self.device.startswith("cuda") else None,
            torch_dtype=torch.bfloat16 if self.device.startswith("cuda") else torch.float32,
            trust_remote_code=True
        )
        self.model.eval()
        self.loaded = True

    def generate(self, prompt, image_path=None, max_new_tokens=256, temperature=0.7, top_p=1.0, top_k=50):
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call `load()` first.")

        content = []
        if self.vision and image_path:
            content.append({"type": "image", "url": image_path})
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k
            )

        response = self.processor.decode(
            outputs[0, inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )

        if "Answer:" in response:
            return response.split("Answer:")[-1].strip()
        return response.strip()"""
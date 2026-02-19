import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .base import BaseLLM

class Llama3LLM(BaseLLM):
    def __init__(self, model_name, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(model_name, vision=False)
        self.device = device
        self.loaded = False

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto" if self.device.startswith("cuda") else None,
            torch_dtype=torch.bfloat16 if self.device.startswith("cuda") else torch.float32,
        )
        self.model.eval()
        self.loaded = True

    def generate(self, prompt_parts, system_instruction=None, max_new_tokens=256, temperature=0.7, top_p=0.9, do_sample=True):
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call `load()` first.")

        if isinstance(prompt_parts, tuple) and len(prompt_parts) == 2:
            system_instruction, blocks = prompt_parts
        else:
            system_instruction = None
            blocks = prompt_parts

        if isinstance(blocks, list):
            user_text = "\n\n".join([p["text"] for p in blocks if p["type"] == "text"])
        else:
            user_text = str(blocks)

        prompt = ""
        if system_instruction:
            prompt += f"System Instruction: {system_instruction}\n\n"
        prompt += f"User: {user_text}\n"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample
            )

        generated_text = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        if "Answer:" in generated_text:
            return generated_text.split("Answer:")[-1].strip()

        return generated_text.strip()
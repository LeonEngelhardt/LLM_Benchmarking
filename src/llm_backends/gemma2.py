import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .base import BaseLLM

class Gemma2LLM(BaseLLM):
    def __init__(self, model_name, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(model_name, vision=False)
        self.device = device
        self.loaded = False

    def load(self):
        """self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto" if self.device.startswith("cuda") else None,
            torch_dtype=torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        )
        self.model.eval()"""
        self.loaded = True

    def generate(self, prompt_parts, max_new_tokens=256, temperature=0.7, do_sample=False):
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call `load()` first.")

        if not isinstance(prompt_parts, tuple) or len(prompt_parts) != 2:
            raise ValueError("prompt_parts must be a tuple: (instruction, blocks)")

        instruction, blocks = prompt_parts

        if isinstance(blocks, list):
            user_text = "\n\n".join([b["text"] for b in blocks if b["type"] == "text"])
        else:
            user_text = str(blocks)

        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": user_text}
        ]

        print(messages)

        """prompt = self.tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                do_sample=do_sample
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "Answer:" in generated_text:
            return generated_text.split("Answer:")[-1].strip()
        return generated_text.strip()"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .base import BaseLLM

class Llama3_1LLM(BaseLLM):
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

    def generate(self, prompt_parts, max_new_tokens=256, temperature=0.7, do_sample=True):
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call `load()` first.")

        
        system_instruction, blocks = prompt_parts

        
        if isinstance(blocks, list):
            user_text = "\n\n".join([p["text"] for p in blocks if p["type"] == "text"])
        else:
            user_text = str(blocks)

        
        messages = []
        
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
            
        messages.append({"role": "user", "content": user_text})

        
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                eos_token_id=terminators,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated_tokens = outputs[0][input_ids.shape[-1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        if "Answer:" in response:
            return response.split("Answer:")[-1].strip()
        return response
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .base import BaseLLM


class MistralLLM(BaseLLM):
    def __init__(self, model_name, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(model_name, vision=False)
        self.device = device
        self.loaded = False

    def load(self):
        use_cuda = torch.cuda.is_available() and "cuda" in self.device
        target_device = "cuda" if use_cuda else "cpu"

        print("torch version:", torch.__version__)
        print("torch cuda runtime:", torch.version.cuda)
        print("cuda available:", torch.cuda.is_available())
        print("cuda device count:", torch.cuda.device_count())
        if torch.cuda.is_available():
            print("device 0:", torch.cuda.get_device_name(0))

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            model_max_length=2048        # ← fix: set here at load time
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16 if use_cuda else torch.float32,
            device_map="auto" if use_cuda else None,
            trust_remote_code=True
        )

        if not use_cuda:
            self.model = self.model.to(target_device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

        self.model.eval()
        self.loaded = True

    def generate(self, prompt_parts, image_paths=None, max_new_tokens=512, temperature=0.7):
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call `load()` first.")

        if isinstance(prompt_parts, tuple) and len(prompt_parts) == 2:
            instruction, blocks = prompt_parts
            system_message = {"role": "system", "content": instruction}
        else:
            system_message = None
            blocks = prompt_parts

        if isinstance(blocks, list):
            text_blocks = [p["text"] for p in blocks if isinstance(p, dict) and p.get("type") == "text"]
            user_content = "\n\n".join(text_blocks)
        else:
            user_content = str(blocks)

        user_message = {"role": "user", "content": user_content}
        messages = [m for m in (system_message, user_message) if m is not None]

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )

        encodings = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )

        if torch.cuda.is_available() and "cuda" in self.device:
            first_device = next(self.model.parameters()).device
            inputs = {k: v.to(first_device) for k, v in encodings.items()}
        else:
            inputs = {k: v.to("cpu") for k, v in encodings.items()}

        print("tokenizer max length:", getattr(self.tokenizer, "model_max_length", "unknown"))
        print("actual input length:", inputs["input_ids"].shape[-1])

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        input_length = inputs["input_ids"].shape[-1]
        generated_tokens = outputs[0][input_length:]

        response = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        ).strip()

        return response
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModel
from src.llm_backends.base import BaseLLM


class InternS1LLM(BaseLLM):
    def __init__(
        self,
        model_name="internlm/Intern-S1",
        vision=True,
        device="cuda",
        verbose=False,
    ):
        super().__init__(model_name, vision=vision, verbose=verbose)

        self.device = device
        self.tokenizer = None
        self.model = None

    def load(self):
        if self.verbose:
            print(f"[Intern-S1] Loading {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)

        self.model.eval()

    @torch.no_grad()
    def generate(self, prompt: str, image_path=None, **kwargs) -> str:
        if self.verbose:
            print("===== Intern-S1 Prompt =====")
            print(prompt)
            print("============================")

        # TEXT ONLY
        if not (self.vision and image_path):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )

            return self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            ).strip()

        # IMAGE + TEXT
        image = Image.open(image_path).convert("RGB")

        response = self.model.chat(
            self.tokenizer,
            image=image,
            question=prompt,
            generation_config=dict(
                max_new_tokens=256,
                do_sample=False,
            ),
        )

        return response.strip()

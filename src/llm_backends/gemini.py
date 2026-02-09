import os
from google import genai
from src.llm_backends.base import BaseLLM


class GeminiLLM(BaseLLM):
    def __init__(self, model_name="gemini-2.5-pro", vision=False, verbose=False):
        super().__init__(model_name, vision=vision, verbose=verbose)

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")

        self.client = genai.Client(api_key=api_key)

    def load(self):
        pass

    def generate(self, prompt: str, image_path=None, **kwargs) -> str:
        if self.verbose:
            print("===== Gemini Prompt =====")
            print(prompt)
            print("=========================")

        # Gemini Vision
        if self.vision and image_path:
            from PIL import Image
            image = Image.open(image_path)

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt, image]
            )
        else:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )

        return response.text.strip()

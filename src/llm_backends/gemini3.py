import os
import requests
import torch
from google import genai
from google.genai import types
from .base import BaseLLM


class Gemini3ProLLM(BaseLLM):
    def __init__(self, model_name):
        super().__init__(model_name, vision=True)
        self.thinking_level = "high"
        self.loaded = False

    def load(self):
        print(f"Loading Gemini model '{self.model_name}'...")
        self.client = genai.Client(
            api_key=os.environ.get("GEMINI_API_KEY")
        )
        self.loaded = True
        print("Gemini client initialized.")

    def generate(self, prompt, image_path):
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call `load()` first.")

        contents = []

        if image_path:
            image_bytes = requests.get(image_path).content

            image_part = types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/jpeg"
            )

            contents.append(image_part)

        contents.append(prompt)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_level=self.thinking_level
                ),
            ),
        )

        return response.text.strip()
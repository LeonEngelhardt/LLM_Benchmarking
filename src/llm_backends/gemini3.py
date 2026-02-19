import os
import requests
from google import genai
from google.genai import types
from PIL import Image
import io
from .base import BaseLLM


class Gemini3ProLLM(BaseLLM):
    def __init__(self, model_name):
        super().__init__(model_name, vision=True)
        self.loaded = False

    def load(self):
        print(f"Loading Gemini model '{self.model_name}'...")
        self.client = genai.Client(
            api_key=os.environ.get("GEMINI_API_KEY")
        )
        self.loaded = True
        print("Gemini client initialized.")

    def _load_image_part(self, image_path):
        try:
            if image_path.startswith("http"):
                image_bytes = requests.get(image_path).content
            else:
                with open(image_path, "rb") as f:
                    image_bytes = f.read()

            return types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/jpeg"
            )
        except Exception as e:
            print(f"[Gemini] Failed to load image {image_path}: {e}")
            return None

    def generate(self, prompt_parts, max_output_tokens=512):

        if not self.loaded:
            raise RuntimeError("Model not loaded. Call `load()` first.")

        if not isinstance(prompt_parts, tuple) or len(prompt_parts) != 2:
            raise ValueError("prompt_parts must be a tuple: (instruction, blocks)")

        instruction, blocks = prompt_parts

        contents = []

        if instruction:
            contents.append(instruction)

        for part in blocks:

            if part["type"] == "text":
                contents.append(part["text"])

            elif part["type"] == "image":
                image_url = part["source"]["url"]
                image_part = self._load_image_part(image_url)
                if image_part:
                    contents.append(image_part)

        print(contents)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                max_output_tokens=max_output_tokens
            ),
        )

        text = response.text.strip()

        if "Answer:" in text:
            return text.split("Answer:")[-1].strip()

        return text






"""import os
import requests
from google import genai
from google.genai import types
from PIL import Image
import io
from .base import BaseLLM


class Gemini3ProLLM(BaseLLM):
    def __init__(self, model_name):
        super().__init__(model_name)
        self.thinking_level = "high"
        self.loaded = False

    def load(self):
        print(f"Loading Gemini model '{self.model_name}'...")
        self.client = genai.Client(
            api_key=os.environ.get("GEMINI_API_KEY")
        )
        self.loaded = True
        print("Gemini client initialized.")

    def _create_dummy_image_part(self):
        img = Image.new("RGB", (1, 1), color="white")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        return types.Part.from_bytes(
            data=buffer.getvalue(),
            mime_type="image/jpeg"
        )

    def _load_image_part(self, image_path):
        if not image_path:
            return self._create_dummy_image_part()

        try:
            if image_path.startswith("http"):
                image_bytes = requests.get(image_path).content
            else:
                with open(image_path, "rb") as f:
                    image_bytes = f.read()

            return types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/jpeg"
            )
        except Exception:
            return self._create_dummy_image_part()

    def generate(self, prompt, image_path=None, examples=None):
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call `load()` first.")

        contents = []

        if examples:
            for ex in examples:
                if ex.get("text"):
                    contents.append(ex["text"])
                if ex.get("image"):
                    img_part = self._load_image_part(ex.get("image"))
                    if img_part:
                        contents.append(img_part)

        if prompt:
            contents.append(prompt)
        if image_path:
            target_img_part = self._load_image_part(image_path)
            if target_img_part:
                contents.append(target_img_part)

        print(contents)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_level=self.thinking_level
                ),
            ),
        )

        return response.text.strip()"""
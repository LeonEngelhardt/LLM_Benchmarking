import os
import requests
from .base import BaseLLM


class OpenRouterLLM(BaseLLM):
    def __init__(self, model_name, vision=False):
        super().__init__(model_name, vision)

    def load(self):
        pass

    def generate(self, prompt_parts, image_paths=None, max_tokens=512, temperature=0.7):

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json"
        }

        if isinstance(prompt_parts, list):
            text_blocks = [p["text"] for p in prompt_parts if p["type"] == "text"]
            prompt_text = "\n\n".join(text_blocks)
        else:
            prompt_text = str(prompt_parts)

        content = []

        if self.vision and image_paths:
            if not isinstance(image_paths, list):
                image_paths = [image_paths]

            for img in image_paths:
                if img:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": img}
                    })

        elif self.vision and isinstance(prompt_parts, list):
            for part in prompt_parts:
                if part["type"] == "image":
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": part["source"]["url"]}
                    })

        content.append({
            "type": "text",
            "text": prompt_text
        })

        if self.vision and len(content) > 1:
            messages = [{
                "role": "user",
                "content": content
            }]
        else:
            messages = [{
                "role": "user",
                "content": prompt_text
            }]

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        res = requests.post(url, headers=headers, json=payload)
        data = res.json()

        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content")

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, dict) and "text" in content:
            return content["text"].strip()

        return str(data)



"""import os, requests
from .base import BaseLLM

class OpenRouterLLM(BaseLLM):
    def __init__(self, model_name):
        super().__init__(model_name)

    def load(self): pass

    """"""
    def generate(self, prompt, image_path=None):
        res = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            },
            json={
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        return res.json()["choices"][0]["message"]["content"]""""""


    def generate(self, prompt, image_path=None):
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json"
        }

        messages = [{"role": "user", "content": prompt}]

        payload = {
            "model": self.model_name,
            "messages": messages
        }

        if image_path:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_path}}
                    ]
                }
            ]
            payload = {
                "model": self.model_name,
                "modalities": ["image", "text"],
                "messages": messages
            }

        res = requests.post(url, headers=headers, json=payload)
        data = res.json()

        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})

        content = message.get("content")
        if isinstance(content, str):
            return content

        if isinstance(content, dict) and "text" in content:
            return content["text"]

        return str(data)"""
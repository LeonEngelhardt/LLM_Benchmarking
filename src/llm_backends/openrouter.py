import os, requests
from .base import BaseLLM

class OpenRouterLLM(BaseLLM):
    def __init__(self, model_name):
        super().__init__(model_name)

    def load(self): pass

    """
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
        return res.json()["choices"][0]["message"]["content"]"""


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

        return str(data)
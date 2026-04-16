from abc import ABC, abstractmethod
from typing import Optional

class BaseLLM(ABC):
    def __init__(self, model_name: str, vision: bool = False):
        self.model_name = model_name
        self.vision = vision
        self.loaded = False

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def generate(self, prompt: str, image_path: Optional[str] = None) -> str:
        pass

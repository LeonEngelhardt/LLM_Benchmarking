from abc import ABC, abstractmethod

class BaseLLM(ABC):
    def __init__(self, model_name: str, vision: bool = False):
        self.model_name = model_name
        self.vision = vision
        self.loaded = False

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def generate(self, prompt: str, image_path: str | None = None) -> str:
        pass


"""
class BaseLLM:
    def __init__(self, model_name, device, hf_token=None):
        self.model_name = model_name
        self.device = device
        self.hf_token = hf_token

    def load_model(self):
        raise NotImplementedError

    def generate(self, prompt: str, image_path=None):
        raise NotImplementedError"""

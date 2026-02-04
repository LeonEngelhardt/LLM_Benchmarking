# api version:
"""
from abc import ABC, abstractmethod

class BaseLLM(ABC):

    def __init__(self, model_name, vision=False, verbose=False):
        self.model_name = model_name
        self.vision = vision
        self.verbose = verbose

    @abstractmethod
    def generate(self, prompt, image_path=None):
        pass"""


class BaseLLM:
    def __init__(self, model_name, device, hf_token=None):
        self.model_name = model_name
        self.device = device
        self.hf_token = hf_token

    def load_model(self):
        raise NotImplementedError

    def generate(self, prompt: str, image_path=None):
        raise NotImplementedError

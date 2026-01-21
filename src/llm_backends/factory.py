from .hf_text import HFTextLLM
from .blip import BlipLLM
from .llama2 import Llama2LLM
from .deepseek_vl import DeepSeekVLLM

def create_llm(model_name, vision, device, hf_token):
    name = model_name.lower()

    if vision:
        if "blip" in name:
            return BlipLLM(model_name, device, hf_token)
        if "deepseek-vl" in name:
            return DeepSeekVLLM(model_name, device, hf_token)
        else:
            raise ValueError(f"Unknown vision model: {model_name}")
    else:
        if "llama" in name:
            return Llama2LLM(model_name, device, hf_token)
        else:
            return HFTextLLM(model_name, device, hf_token)

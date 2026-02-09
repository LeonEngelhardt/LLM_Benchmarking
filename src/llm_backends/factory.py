from .hf_text import HFTextLLM
from .blip import BlipLLM
from .deepseek_vl import DeepSeekVLLM
from .qwen_vl import QwenVLLM

from .openai import OpenAILLM
from .openrouter import OpenRouterLLM
from .anthropic import AnthropicLLM
from .gemini import GeminiLLM

def get_llm(model_name: str, vision: bool, device="cpu", hf_token=None):
    name = model_name.lower()

    # ---------- API MODELS ----------
    #if name.startswith("gpt") or "openai" in name:
    #    return OpenAILLM(model_name)

    if "openrouter" in name:
        return OpenRouterLLM(model_name)

    if "claude" in name:
        return AnthropicLLM(model_name)

    if "gemini" in name:
        return GeminiLLM(model_name, vision=vision)

    # ---------- LOCAL HF MODELS ----------
    if model_name == "gpt2" or "llama" in name or "mistral" in name:
        return HFTextLLM(model_name, device, hf_token)

    if "blip" in name:
        return BlipLLM(model_name, device)

    if "deepseek" in name:
        return DeepSeekVLLM(model_name, device)

    if "qwen-vl" in name:
        return QwenVLLM(model_name, device)

    raise ValueError(f"Unknown model backend: {model_name}")


"""
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
            """
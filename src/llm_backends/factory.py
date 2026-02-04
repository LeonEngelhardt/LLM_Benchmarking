# api version:
"""from .openai import OpenAIBackend
from .anthropic import AnthropicBackend
from .openrouter import OpenRouterBackend
from .qwen import QwenBackend

def get_llm(model_name, vision=False, verbose=False):

    # OpenAI
    if model_name in ["gpt-4o", "gpt-3.5-turbo", "o3-mini", "o4-mini"]:
        return OpenAIBackend(model_name, vision=vision, verbose=verbose)

    # Anthropic
    if model_name in ["claude-3.7-sonnet", "claude-3.5-haiku"]:
        return AnthropicBackend(model_name, vision=vision, verbose=verbose)

    # OpenRouter models
    openrouter_models = [
        "deepseek-r1", "grok-3", "llama-3.3-70b", "llama-3.1-405b",
        "llama-3.1-8b", "gemini-2.5-flash", "gemini-2.5-pro"
    ]
    if model_name in openrouter_models:
        return OpenRouterBackend(model_name, vision=vision, verbose=verbose)

    # Qwen
    if model_name in ["qwen/qwen3-235b", "qwen/qwen3-32b"]:
        return QwenBackend(model_name, vision=vision, verbose=verbose)

    raise ValueError(f"Unsupported model: {model_name}")"""


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
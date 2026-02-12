from .hf_text import HFTextLLM
from .blip import BlipLLM
from .deepseek import DeepSeekV3LLM
from .qwen3 import Qwen3VLLLM
from .openai import OpenAILLM
from .openrouter import OpenRouterLLM
from .anthropic import AnthropicLLM
from .gemini3 import Gemini3ProLLM
from .intern_s1 import InternS1LLM
from .llava import LlavaLLM
from .llama4 import Llama4MultimodalLLM
from .gemma3 import Gemma3MultimodalLLM
from .mistral import MistralLLM


def get_llm(model_name: str, vision: bool, device="cpu", hf_token=None):
    name = model_name.lower()

    # API MODELS
    #if "openai" in name or name.startswith("gpt"):
    #    return OpenAILLM(model_name)

    if "openrouter" in name:
        return OpenRouterLLM(model_name)

    if "claude" in name:
        return AnthropicLLM(model_name)

    if "gemini3" in name:
        return Gemini3ProLLM(model_name, vision=vision)

    if "gemma3" in name:
        return Gemma3MultimodalLLM(model_name, vision=vision)

    # LOCAL HF MODELS - Text only
    if name == "gpt2" or "llama" in name or "mistral" in name:
        return HFTextLLM(model_name, device, hf_token)

    # LOCAL HF MODELS - Vision / Multimodal
    if "llava" in name:
        return LlavaLLM(model_name, device)

    if "llama4" in name:
        return Llama4MultimodalLLM(model_name, device)

    if "mistral" in name:
        return MistralLLM(model_name, device)

    if "blip" in name:
        return BlipLLM(model_name, device)

    if "deepseek" in name:
        return DeepSeekV3LLM(model_name, device)

    if "qwen3" in name or "qwen-vl" in name:
        return Qwen3VLLLM(model_name, device)

    if "intern-s1" in name:
        return InternS1LLM(model_name, device)

    raise ValueError(f"Unknown model backend: {model_name}")




"""from .hf_text import HFTextLLM
from .blip import BlipLLM
from .deepseek import DeepSeekVLLM
from .qwen3 import QwenVLLM
from .openai import OpenAILLM
from .openrouter import OpenRouterLLM
from .anthropic import AnthropicLLM
from .gemini3 import GeminiLLM
from .intern_s1 import InternS1LLM

def get_llm(model_name: str, vision: bool, device="cpu", hf_token=None):
    name = model_name.lower()

    # API MODELS
    #if name.startswith("gpt") or "openai" in name:
    #    return OpenAILLM(model_name)

    if "openrouter" in name:
        return OpenRouterLLM(model_name)

    if "claude" in name:
        return AnthropicLLM(model_name)

    if "gemini" in name:
        return GeminiLLM(model_name, vision=vision)

    # LOCAL HF MODELS
    if model_name == "gpt2" or "llama" in name or "mistral" in name:
        return HFTextLLM(model_name, device, hf_token)

    if "blip" in name:
        return BlipLLM(model_name, device)

    if "deepseek" in name:
        return DeepSeekVLLM(model_name, device)

    if "qwen-vl" in name:
        return QwenVLLM(model_name, device)
    
    if "intern-s1" in model_name.lower():
        return InternS1LLM(model_name, device)

    raise ValueError(f"Unknown model backend: {model_name}")"""
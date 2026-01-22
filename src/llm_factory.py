# local --> can be deleted
from src.llm_backends import TextLLM, BlipVLM

def load_llm(model_name, vision, device, hf_token=None):
    if vision:
        if "blip" in model_name.lower():
            return BlipVLM(model_name, device)
        else:
            raise NotImplementedError(f"Vision model not yet supported: {model_name}")
    else:
        return TextLLM(model_name, device, hf_token)
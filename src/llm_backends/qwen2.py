import torch
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from .base import BaseLLM

def get_qwen_model_class(model_name):
    if "Qwen2.5-VL" in model_name:
        from transformers import Qwen2_5_VLForConditionalGeneration
        return Qwen2_5_VLForConditionalGeneration
    elif "Qwen2-VL" in model_name:
        from transformers import Qwen2VLForConditionalGeneration
        return Qwen2VLForConditionalGeneration
    else:
        raise ValueError(f"Unrecognized Qwen model: {model_name}")

class Qwen2VLLLM(BaseLLM):
    def __init__(self, model_name, device="cuda" if torch.cuda.is_available() else "cpu", vision=True):
        super().__init__(model_name, vision)
        self.device = device
        self.loaded = False

    def load(self):
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        ModelClass = get_qwen_model_class(self.model_name)
        self.model = ModelClass.from_pretrained(
            self.model_name,
            device_map="auto" if self.device.startswith("cuda") else None,
            torch_dtype=torch.bfloat16 if self.device.startswith("cuda") else torch.float32,
            trust_remote_code=True
        )
        self.model.eval()
        self.loaded = True

    def generate(self, prompt_parts, image_paths=None, max_new_tokens=256):

        if not self.loaded:
            raise RuntimeError("Model not loaded. Call `load()` first.")

        if not isinstance(prompt_parts, tuple) or len(prompt_parts) != 2:
            raise ValueError("prompt_parts must be a tuple: (instruction, blocks)")

        instruction, blocks = prompt_parts
        content = []

        if self.vision and image_paths:
            if not isinstance(image_paths, list):
                image_paths = [image_paths]
            for img in image_paths:
                if img:
                    content.append({"type": "image", "image": img})

        elif isinstance(blocks, list):
            for part in blocks:
                if part["type"] == "image":
                    content.append({"type": "image", "image": part["source"]["url"]})

        if isinstance(blocks, list):
            text_blocks = [p["text"] for p in blocks if p["type"] == "text"]
            full_text = "\n\n".join(text_blocks)
        else:
            full_text = str(blocks)

        content.append({"type": "text", "text": full_text})

        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": content}
        ]

        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        print(text_input)

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text_input],
            images=image_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        response = output_text[0].strip()
        if "Answer:" in response:
            return response.split("Answer:")[-1].strip()
        return response
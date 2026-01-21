from transformers import AutoProcessor, AutoModelForSeq2SeqLM

class DeepSeekVLLM:
    def __init__(self, model_name, device="cpu", hf_token=None):
        self.model_name = model_name
        self.device = device
        self.hf_token = hf_token

    def load_model(self):
        self.processor = AutoProcessor.from_pretrained(self.model_name, use_auth_token=self.hf_token)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name, use_auth_token=self.hf_token
        ).to(self.device)

    def generate(self, text_inputs, pixel_values=None):
        inputs = self.processor(
            images=pixel_values,
            text=text_inputs,
            return_tensors="pt"
        ).to(self.device)
        out = self.model.generate(**inputs)
        return self.processor.batch_decode(out, skip_special_tokens=True)[0]

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class Generator:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True
        )
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

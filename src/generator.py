from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

PROMPT_TEMPLATE = """You are an expert assistant.
You are given context extracted from academic PDFs.
Use ONLY this context to answer the question.

Rules:
- Answer in clear, complete sentences.
- Do NOT copy random phrases.
- Do NOT hallucinate.
- If the context does not contain the answer, say:
  "The context does not contain the answer."

Context:
{context}

Question:
{question}

Answer:
"""

class Generator:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Day‑2: new signature (query + retrieved chunks)
    def generate(self, query: str, retrieved_chunks: List[Dict], max_new_tokens: int = 256) -> Dict:

        # If nothing retrieved → no answer
        if not retrieved_chunks:
            return {
                "answer": "The context does not contain the answer.",
                "retrieved_chunks": []
            }

        # Merge multiple chunks into one context block
        context = "\n\n---\n\n".join([c["text"] for c in retrieved_chunks])

        # Build final prompt
        prompt = PROMPT_TEMPLATE.format(context=context, question=query)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        if not answer:
            answer = "The context does not contain the answer."

        return {
            "answer": answer,
            "retrieved_chunks": retrieved_chunks
        }

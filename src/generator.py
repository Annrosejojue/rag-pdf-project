from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .config import MAX_NEW_TOKENS, ENABLE_SELF_CHECK, ENABLE_CITATIONS

PROMPT_TEMPLATE = """You are an expert assistant.
You are given context extracted from academic PDFs.
Use ONLY this context to answer the question.

Rules:
- Answer in clear, complete sentences.
- Do NOT hallucinate.
- If the context does not contain the answer, say:
  "The context does not contain the answer."

Context:
{context}

Question:
{question}

Answer:
"""

SELF_CHECK_TEMPLATE = """You are verifying an answer.

Context:
{context}

Answer:
{answer}

Question:
Does the answer strictly follow the context? Reply only YES or NO.
"""

class Generator:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="auto"
        )

    def _merge_chunks(self, chunks: List[Dict]) -> str:
        return "\n\n---\n\n".join([c["text"] for c in chunks])

    def _self_check(self, context: str, answer: str) -> bool:
        prompt = f"<s>[INST] {SELF_CHECK_TEMPLATE.format(context=context, answer=answer)} [/INST]"
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False
            )

        verdict = self.tokenizer.decode(out[0], skip_special_tokens=True).strip()
        return verdict.upper().startswith("YES")

    def generate(self, query: str, retrieved_chunks: List[Dict]) -> Dict:

        if not retrieved_chunks:
            return {
                "answer": "The context does not contain the answer.",
                "retrieved_chunks": []
            }

        context = self._merge_chunks(retrieved_chunks)

        # Correct chat-style wrapping
        prompt = f"<s>[INST] {PROMPT_TEMPLATE.format(context=context, question=query)} [/INST]"

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
            add_special_tokens=False
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False
            )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        if ENABLE_SELF_CHECK:
            ok = self._self_check(context, answer)
            if not ok:
                answer = "The context does not contain the answer."

        if ENABLE_CITATIONS:
            answer += "\n\nSources:\n"
            for c in retrieved_chunks:
                answer += f"- [{c['doc_id']}#{c['chunk_id']}]\n"

        return {
            "answer": answer,
            "retrieved_chunks": retrieved_chunks
        }

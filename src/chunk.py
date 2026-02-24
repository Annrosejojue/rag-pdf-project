from typing import List, Dict
from .config import CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS

def simple_word_chunk(text: str, max_tokens: int, overlap: int) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

def chunk_documents(docs: List[Dict]) -> List[Dict]:
    all_chunks = []
    for doc in docs:
        chunks = simple_word_chunk(
            doc["text"],
            max_tokens=CHUNK_SIZE_TOKENS,
            overlap=CHUNK_OVERLAP_TOKENS
        )
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "doc_id": doc["id"],
                "chunk_id": i,
                "text": chunk
            })
    return all_chunks

from typing import List, Dict
import numpy as np
from .embed import EmbeddingModel
from .vector_store import FaissVectorStore
from .config import TOP_K

class Retriever:
    def __init__(self, embedder: EmbeddingModel, vector_store: FaissVectorStore):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: str, k: int = TOP_K) -> List[Dict]:
        q_emb = self.embedder.encode([query])  # shape (1, dim)
        q_emb = np.array(q_emb)
        results = self.vector_store.search(q_emb, k=k)
        return results

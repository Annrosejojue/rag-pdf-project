from typing import List, Dict
import numpy as np
from .embed import EmbeddingModel
from .vector_store import FaissVectorStore
from .config import TOP_K
from .config import RERANKER_MODEL_NAME, RETRIEVER_TOP_K, RERANK_TOP_K
from sentence_transformers import CrossEncoder


class Retriever:
    def __init__(self, embedder: EmbeddingModel, vector_store: FaissVectorStore):
        self.embedder = embedder
        self.vector_store = vector_store
        self.reranker = CrossEncoder(RERANKER_MODEL_NAME)

    def retrieve(self, query: str, k: int = TOP_K) -> List[Dict]:
        q_emb = self.embedder.encode([query])  # shape (1, dim)
        q_emb = np.array(q_emb)
        results = self.vector_store.search(q_emb, k=RETRIEVER_TOP_K)
        if not results:
            return []
        pairs = [(query, r["text"]) for r in results]
        scores = self.reranker.predict(pairs)
        scored = list(zip(results, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        reranked = [item for item, _ in scored[:RERANK_TOP_K]]
        return reranked

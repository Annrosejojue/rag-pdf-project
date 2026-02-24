import faiss
import numpy as np
from typing import List, Dict

class FaissVectorStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)
        self.metadata: List[Dict] = []

    def add(self, embeddings: np.ndarray, metadatas: List[Dict]):
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype("float32")
        self.index.add(embeddings)
        self.metadata.extend(metadatas)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype("float32")
        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)

        distances, indices = self.index.search(query_embedding, k)
        results = []
        for idx in indices[0]:
            results.append(self.metadata[idx])
        return results

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class EmbeddingModel:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        return embeddings

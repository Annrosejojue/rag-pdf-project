from typing import List, Dict
from .config import EMBEDDING_MODEL_NAME, GENERATOR_MODEL_NAME, TOP_K
from .embed import EmbeddingModel
from .vector_store import FaissVectorStore
from .retriever import Retriever
from .generator import Generator

class RAGPipeline:
    def __init__(self, embedding_model_name: str = EMBEDDING_MODEL_NAME,
                 generator_model_name: str = GENERATOR_MODEL_NAME,
                 embedding_dim: int = 384):
        # 384 is the dimension for all-MiniLM-L6-v2
        self.embedder = EmbeddingModel(embedding_model_name)
        self.vector_store = FaissVectorStore(embedding_dim)
        self.retriever = Retriever(self.embedder, self.vector_store)
        self.generator = Generator(generator_model_name)

    def index_chunks(self, chunks_with_meta: List[Dict]):
        texts = [c["text"] for c in chunks_with_meta]
        embeddings = self.embedder.encode(texts)
        self.vector_store.add(embeddings, chunks_with_meta)

    def answer(self, query: str, k: int = TOP_K) -> Dict:
        retrieved = self.retriever.retrieve(query, k=k)
        context = "\n\n".join([f"[{r['doc_id']}#{r['chunk_id']}]\n{r['text']}" for r in retrieved])

        prompt = (
            "You are a helpful assistant. Use ONLY the context below to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )

        answer_text = self.generator.generate(prompt)
        return {
            "answer": answer_text,
            "retrieved_chunks": retrieved
        }

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GENERATOR_MODEL_NAME = "google/flan-t5-xl"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

PDF_DIR = "data/pdfs"

# Chunking
CHUNK_SIZE_TOKENS = 256
CHUNK_OVERLAP_TOKENS = 32

# Retrieval
TOP_K = 5
RETRIEVER_TOP_K = 8
RERANK_TOP_K = 3

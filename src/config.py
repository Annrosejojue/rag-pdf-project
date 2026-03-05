# Embedding model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GENERATOR_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# Retrieval settings
TOP_K = 3
RETRIEVER_TOP_K = 8
RERANK_TOP_K = 3

# Generation settings
MAX_NEW_TOKENS = 512
MAX_CONTEXT_CHARS = 8000   # prevent overloading Llama
ENABLE_SELF_CHECK = True
ENABLE_CITATIONS = True
CHUNK_SIZE_TOKENS = 300
CHUNK_OVERLAP_TOKENS = 50
PDF_DIR = "data/pdfs"

from src.ingest import load_pdfs
from src.chunk import chunk_documents
from src.rag_pipeline import RAGPipeline

def build_index():
    docs = load_pdfs()
    chunks = chunk_documents(docs)

    rag = RAGPipeline(
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        generator_model_name="mistralai/Mistral-7B-Instruct-v0.2",
        embedding_dim=384
    )

    rag.index_chunks(chunks)
    return rag

if __name__ == "__main__":
    rag = build_index()

    while True:
        query = input("Query (empty to exit): ")
        if not query:
            break

        result = rag.answer(query)
        print("\nAnswer:\n", result["answer"])

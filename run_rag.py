from src.ingest import load_pdfs
from src.chunk import chunk_documents
from src.rag_pipeline import RAGPipeline

def build_index() -> RAGPipeline:
    print("Loading PDFs...")
    docs = load_pdfs()
    print(f"Loaded {len(docs)} documents.")

    print("Chunking documents...")
    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks.")

    print("Initializing RAG pipeline...")
    rag = RAGPipeline(
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        generator_model_name="google/flan-t5-base",
        embedding_dim=384
    )

    print("Indexing chunks into vector store...")
    rag.index_chunks(chunks)
    print("Indexing complete.")

    return rag

if __name__ == "__main__":
    rag = build_index()
    print("\nRAG system is ready. Ask questions about your PDFs.\n")

    while True:
        query = input("Query (empty to exit): ").strip()
        if not query:
            print("Goodbye.")
            break

        result = rag.answer(query)
        print("\nAnswer:\n", result["answer"])
        print("\n(Top retrieved chunks were from these doc IDs and chunk IDs):")
        for r in result["retrieved_chunks"]:
            print(f" - {r['doc_id']}#{r['chunk_id']}")
        print("\n" + "-"*60 + "\n")

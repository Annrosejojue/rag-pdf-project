Day 1
We used MiniLM (all‑MiniLM‑L6‑v2) as the embedding model to convert text into vectors.
We used FLAN‑T5‑Base as the generator model to produce answers.
The embedding model worked perfectly — retrieval was accurate.
The generator model was too small, so answers were short, random, or incomplete.
Day 2 Quality Upgrades
Added Cross‑Encoder reranker (ms-marco-MiniLM-L-6-v2)
Added multi‑chunk context merging
Added strong prompt template
Added no‑answer detection Upgraded generator to support (query, retrieved_chunks)
Day 3 adds a stronger generator model, dynamic chunk‑windowing, and self‑verification to improve answer accuracy and reduce hallucinations.  
The pipeline now uses an instruction‑tuned causal LM  first tried using meta-llama then moved to (Mistral‑7B) with merged context, reranking, and citation‑aware output.  

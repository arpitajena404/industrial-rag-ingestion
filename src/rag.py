"""
rag.py
─────────────────────────────────────────────────────────────────────────────
This module implements the core RAG (Retrieval-Augmented Generation) pipeline:
1. Converts the user's natural language question into a semantic vector.
2. Connects to our local persistent ChromaDB to perform a vector search.
3. Retrieves the top-K matching document chunks (with original metadata).
4. Formulates a system prompt injecting the retrieved context chunks.
5. Sends the combined prompt to Groq (llama-3.3-70b) for synthesis.
6. Returns the human-like summarized answer along with original source citations.
"""

import os
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv
from groq import Groq

# 1. LOAD CONFIGURATION
dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.env"))
load_dotenv(dotenv_path=dotenv_path)

# Paths and variables matching our ingestion pipelines
CHROMA_DIR      = os.path.abspath(os.path.join(os.path.dirname(__file__), "../chroma_db"))
COLLECTION_NAME = "industrial_knowledge"
EMBED_MODEL     = "all-MiniLM-L6-v2"

# Local sentence transformer instance (loaded lazily on the first function call)
_model = None

def get_embedding_model():
    """Lazily instantiates and caches the SentenceTransformer model to save memory."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model

def retrieve_context(query: str, top_k: int = 5):
    """
    Retrieval Component
    - Vectorizes the user query.
    - Connects to ChromaDB.
    - Searches for chunks closest in the multi-dimensional vector space.
    """
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME not in existing:
        raise ValueError(
            f"Collection '{COLLECTION_NAME}' not found in ChromaDB at {CHROMA_DIR}.\n"
            "Please run 'python src/embed.py' first to populate vectors."
        )

    collection = client.get_collection(COLLECTION_NAME)

    model = get_embedding_model()
    query_embedding = model.encode([query])[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    retrieved_docs = []
    if results and "documents" in results and results["documents"]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            retrieved_docs.append({
                "text"    : doc,
                "metadata": meta,
                "score"   : round(1.0 - dist, 4)
            })

    return retrieved_docs

def generate_answer(query: str, top_k: int = 5):
    """
    Augmentation and Generation Component
    - Fetches the vector search context from ChromaDB.
    - Injects it into the system prompt.
    - Calls Groq llama-3.3-70b to generate a cited response.
    """
    # 1. Fetch relevant chunks
    docs = retrieve_context(query, top_k=top_k)

    # 2. Format context string
    context_str = ""
    for i, doc in enumerate(docs, 1):
        meta = doc["metadata"]
        context_str += f"--- Document {i} (Source: {meta.get('source_file')}, Section: {meta.get('section', 'General')}) ---\n"
        context_str += f"{doc['text']}\n\n"

    # 3. Call Groq
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=1000,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert assistant specialized in industrial plants and operations.\n"
                    "Use the provided document context below to answer the user's question. "
                    "If the context doesn't contain the answer, say you don't know based on the documents.\n\n"
                    "=== Retrieved Document Context ===\n"
                    f"{context_str}"
                    "==================================\n\n"
                    "Generate a clear, professional, and accurate response. "
                    "Always cite the source files and sections where you found the information."
                )
            },
            {
                "role": "user",
                "content": query
            }
        ]
    )

    answer = response.choices[0].message.content

    return {
        "answer" : answer,
        "sources": docs
    }

# LOCAL TERMINAL EXECUTION
if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "What corrective actions were taken after the dry running incident?"

    print(f"Query: {query}\n")
    print("Retrieving context and generating answer...")
    try:
        res = generate_answer(query)
        print("\n=== Answer ===")
        print(res["answer"])
        print("\n=== Sources Cited ===")
        for idx, src in enumerate(res["sources"], 1):
            meta = src["metadata"]
            print(f"[{idx}] {meta.get('source_file')} | Section: {meta.get('section')} (Similarity: {src['score']:.4f})")
    except Exception as e:
        print(f"Error during execution: {e}")
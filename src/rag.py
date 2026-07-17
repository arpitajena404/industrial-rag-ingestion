"""
rag.py
─────────────────────────────────────────────────────────────────────────────
This module implements the core RAG (Retrieval-Augmented Generation) pipeline:
1. Converts the user's natural language question into a semantic vector.
2. Connects to our local persistent ChromaDB to perform a vector search.
3. Detects an equipment tag in the query (e.g. HX-301) and additionally
   pulls any chunk whose TEXT contains that tag, so multi-section documents
   don't lose sections that scored low semantically for this specific question.
4. Retrieves the top-K matching document chunks (with original metadata).
5. Applies feedback-based score adjustments from past thumbs up/down votes.
6. Formulates a system prompt injecting the retrieved context chunks.
7. Sends the combined prompt to Groq (llama-3.3-70b) for synthesis.
8. Returns the human-like summarized answer along with original source citations.
"""

import os
import re
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv
from groq import Groq
from feedback import get_chunk_adjustments

dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.env"))
load_dotenv(dotenv_path=dotenv_path)

CHROMA_DIR      = os.path.abspath(os.path.join(os.path.dirname(__file__), "../chroma_db"))
COLLECTION_NAME = "industrial_knowledge"
EMBED_MODEL     = "all-MiniLM-L6-v2"

# Generalized equipment tag pattern — matches HX-301, PUMP-101, TANK-201,
# VFD-101, MTR-101, CT-101, etc. Used only to detect a tag IN THE QUERY;
# does not touch stored metadata, so it works immediately on existing data.
EQUIPMENT_TAG_QUERY_PATTERN = re.compile(r"\b[A-Za-z]{2,6}[-\s]?\d{2,4}\b")

# Cap how many tag-matched chunks we force into context — this document
# alone has ~14 chunks containing "HX-301", and pulling all of them would
# both blow up the prompt size and burn through the daily Groq token quota
# faster. 6 is enough to cover most sections of a single work order.
MAX_TAG_MATCHES = 6

_model = None

def get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model

def detect_equipment_tag(query: str):
    match = EQUIPMENT_TAG_QUERY_PATTERN.search(query)
    if not match:
        return None
    return match.group(0).upper().replace(" ", "-")

def retrieve_context(query: str, top_k: int = 5):
    """
    Retrieval Component
    - Vectorizes the user query, does semantic search.
    - If an equipment tag is detected in the query, ALSO pulls any chunk
      whose raw text literally contains that tag (via ChromaDB's
      where_document contains filter), so sections that scored low
      semantically for this exact phrasing still get included.
    - Merges both, applies feedback adjustments, re-sorts, trims to top_k
      (plus the guaranteed tag matches).
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

    fetch_k = top_k + 5

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=fetch_k,
        include=["documents", "metadatas", "distances"]
    )

    candidates = {}
    if results and "documents" in results and results["documents"]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            chunk_id = meta.get("chunk_id", doc[:30])
            candidates[chunk_id] = {
                "text"    : doc,
                "metadata": meta,
                "score"   : round(1.0 - dist, 4)
            }

    # ── Equipment-tag guarantee: pull chunks whose TEXT contains the tag ──
    tag = detect_equipment_tag(query)
    if tag:
        try:
            tag_results = collection.get(
                where_document={"$contains": tag},
                include=["documents", "metadatas"],
                limit=MAX_TAG_MATCHES
            )
            for doc, meta in zip(tag_results.get("documents", []), tag_results.get("metadatas", [])):
                chunk_id = meta.get("chunk_id", doc[:30])
                if chunk_id not in candidates:
                    # Not already in semantic results — force it in with a
                    # high score so it survives the final sort/trim below.
                    candidates[chunk_id] = {
                        "text"    : doc,
                        "metadata": meta,
                        "score"   : 0.99
                    }
        except Exception as e:
            # If where_document isn't supported by this Chroma version,
            # don't crash the whole query over this optional enhancement.
            print(f"Tag-match lookup skipped: {e}")

    candidate_list = list(candidates.values())

    adjustments = get_chunk_adjustments()
    for c in candidate_list:
        chunk_id = c["metadata"].get("chunk_id", "")
        c["score"] = round(c["score"] + adjustments.get(chunk_id, 0), 4)

    candidate_list.sort(key=lambda c: c["score"], reverse=True)

    final_k = top_k + (MAX_TAG_MATCHES if tag else 0)
    return candidate_list[:final_k]

def generate_answer(query: str, top_k: int = 5):
    docs = retrieve_context(query, top_k=top_k)

    context_str = ""
    for i, doc in enumerate(docs, 1):
        meta = doc["metadata"]
        context_str += f"--- Document {i} (Source: {meta.get('source_file')}, Section: {meta.get('section', 'General')}) ---\n"
        context_str += f"{doc['text']}\n\n"

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
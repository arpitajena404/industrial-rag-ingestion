"""
embed.py
────────────────────────────────────────────────────────
Person B's script.

What this does:
  1. Reads chunks.jsonl (output from Person A)
  2. Filters out any junk chunks (too short)
  3. Generates embeddings using sentence-transformers (free, runs locally)
  4. Stores everything in ChromaDB with full metadata
  5. Runs a quick sanity-check query at the end to confirm it works

How to run (full rebuild — deletes and re-embeds everything):
  pip install chromadb sentence-transformers tqdm
  python embed.py

For incremental adds (used by ingest_new_document.py), import
embed_new_chunks() instead — it does NOT delete the collection.

Output:
  A ChromaDB database folder called `chroma_db/` in your project root.
  This folder is what Person D's chat interface will query.
"""

import json
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# ──────────────────────────────────────────────────────
# CONFIG — change these paths if needed
# ──────────────────────────────────────────────────────

CHUNKS_FILE   = "../data/processed/chunks.jsonl"   # output from Person A
CHROMA_DIR    = "../chroma_db"                      # where ChromaDB will be stored
COLLECTION    = "industrial_knowledge"           # name of the collection
EMBED_MODEL   = "all-MiniLM-L6-v2"              # free, fast, good quality
BATCH_SIZE    = 64                               # how many chunks to embed at once
MIN_CHARS     = 50                               # skip chunks shorter than this


def build_metadata(chunk: dict) -> dict:
    """
    ChromaDB metadata must be: str, int, float, or bool only.
    No lists, no None values allowed.
    This function sanitises the chunk fields before storing.
    """
    return {
        # Core identifiers
        "chunk_id"      : chunk.get("chunk_id", ""),
        "doc_id"        : chunk.get("doc_id", ""),
        "source_file"   : chunk.get("source_file", ""),
        "file_type"     : chunk.get("file_type", ""),
        "doc_type"      : chunk.get("doc_type", ""),

        # Content metadata
        "section"       : chunk.get("section", "") or "",
        "chunk_index"   : int(chunk.get("chunk_index", 0)),
        "char_count"    : int(chunk.get("char_count", 0)),

        # equipment_tags is a LIST — join to comma-separated string
        # e.g. ["PUMP-101", "PUMP-102"] → "PUMP-101,PUMP-102"
        "equipment_tags": ",".join(chunk.get("equipment_tags") or []),

        # Dates — None becomes empty string
        "filename_date" : chunk.get("filename_date") or "",
        "ingested_at"   : chunk.get("ingested_at", ""),

        # Quality flags
        "needs_review"      : bool(chunk.get("needs_review", False)),
        "header_confident"  : bool(chunk.get("header_confident", True)),

        # Sheet info (for Excel-sourced chunks)
        "sheet_name"    : chunk.get("sheet_name") or "",
    }


# ──────────────────────────────────────────────────────
# Incremental path — used by ingest_new_document.py
# Adds ONLY the given chunks to the existing collection.
# Never deletes anything. Safe to call repeatedly.
# ──────────────────────────────────────────────────────

_cached_model = None

def get_cached_model():
    """Loads the SentenceTransformer model once and reuses it across calls —
    avoids reloading the ~90MB model on every single ingest."""
    global _cached_model
    if _cached_model is None:
        _cached_model = SentenceTransformer(EMBED_MODEL)
    return _cached_model


def embed_new_chunks(new_chunks: list) -> int:
    """
    Embeds and adds ONLY new_chunks to the existing ChromaDB collection.
    Does NOT delete or rebuild — safe to call after every new document.
    Returns the number of chunks actually embedded (after MIN_CHARS filtering).
    """
    if not new_chunks:
        return 0

    new_chunks = [c for c in new_chunks if len(c["text"]) >= MIN_CHARS]
    if not new_chunks:
        return 0

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )

    model = get_cached_model()

    texts = [c["text"] for c in new_chunks]
    embeddings = model.encode(texts, show_progress_bar=False)

    collection.add(
        ids=[c["chunk_id"] for c in new_chunks],
        embeddings=embeddings.tolist(),
        metadatas=[build_metadata(c) for c in new_chunks],
        documents=texts,
    )
    return len(new_chunks)


# ──────────────────────────────────────────────────────
# Full rebuild path — ONLY runs when you execute this file
# directly (python embed.py). Never runs on import, so
# ingest_new_document.py importing embed_new_chunks above
# will NOT trigger this and wipe your data.
# ──────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── STEP 1 — Load chunks ──────────────────────────
    print("=" * 55)
    print("  STEP 1: Loading chunks")
    print("=" * 55)

    chunks = []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    print(f"  Total chunks loaded : {len(chunks)}")

    chunks = [c for c in chunks if len(c["text"]) >= MIN_CHARS]
    print(f"  After filtering     : {len(chunks)} chunks (removed chunks < {MIN_CHARS} chars)")

    from collections import Counter
    type_counts = Counter(c["doc_type"] for c in chunks)
    print("\n  Breakdown by doc_type:")
    for dt, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {dt:<45} {count} chunks")

    # ── STEP 2 — Load the embedding model ─────────────
    print("\n" + "=" * 55)
    print("  STEP 2: Loading embedding model")
    print("=" * 55)

    print(f"  Model: {EMBED_MODEL}")
    print("  (First run downloads ~90MB — subsequent runs use cache)")

    model = get_cached_model()
    print("  Model loaded.")

    # ── STEP 3 — Set up ChromaDB ───────────────────────
    print("\n" + "=" * 55)
    print("  STEP 3: Setting up ChromaDB")
    print("=" * 55)

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    existing = [c.name for c in client.list_collections()]
    if COLLECTION in existing:
        print(f"  Deleting existing collection '{COLLECTION}' for clean re-run...")
        client.delete_collection(COLLECTION)

    collection = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )
    print(f"  Collection '{COLLECTION}' created.")

    # ── STEP 4 — Embed and store in batches ───────────
    print("\n" + "=" * 55)
    print("  STEP 4: Embedding chunks and storing in ChromaDB")
    print("=" * 55)

    total = len(chunks)
    for batch_start in tqdm(range(0, total, BATCH_SIZE), desc="  Embedding batches"):
        batch = chunks[batch_start : batch_start + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        embeddings = model.encode(texts, show_progress_bar=False)

        ids        = [c["chunk_id"] for c in batch]
        metadatas  = [build_metadata(c) for c in batch]
        documents  = texts

        collection.add(
            ids        = ids,
            embeddings = embeddings.tolist(),
            metadatas  = metadatas,
            documents  = documents,
        )

    print(f"\n  Done. {collection.count()} chunks stored in ChromaDB.")

    # ── STEP 5 — Sanity check: run 3 test queries ─────
    print("\n" + "=" * 55)
    print("  STEP 5: Sanity check — test queries")
    print("=" * 55)

    test_queries = [
        "What was the vibration level on PUMP-101 before the bearing failed?",
        "Which vendor is nearest to Bharat Steel Plant?",
        "What corrective actions were taken after the dry running incident?",
    ]

    for query in test_queries:
        print(f"\n  Query: \"{query}\"")
        results = collection.query(
            query_texts   = [query],
            n_results     = 3,
            include       = ["documents", "metadatas", "distances"]
        )
        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )):
            score = round(1 - dist, 3)
            print(f"    [{i+1}] Score={score} | {meta['source_file']} | {meta['doc_type']}")
            print(f"         \"{doc[:120]}...\"")

    # ── STEP 6 — Filtered query example ───────────────
    print("\n" + "=" * 55)
    print("  STEP 6: Filtered query example")
    print("  (Only work_order chunks mentioning PUMP-101)")
    print("=" * 55)

    filtered = collection.query(
        query_texts    = ["bearing replacement maintenance"],
        n_results      = 3,
        where          = {
            "$and": [
                {"doc_type"      : {"$eq": "work_order"}},
                {"equipment_tags": {"$contains": "PUMP-101"}},
            ]
        },
        include        = ["documents", "metadatas", "distances"]
    )

    for i, (doc, meta, dist) in enumerate(zip(
        filtered["documents"][0],
        filtered["metadatas"][0],
        filtered["distances"][0]
    )):
        score = round(1 - dist, 3)
        print(f"  [{i+1}] Score={score} | {meta['source_file']}")
        print(f"       \"{doc[:150]}...\"")

    print("\n" + "=" * 55)
    print("  ALL DONE.")
    print(f"  ChromaDB stored at: ./{CHROMA_DIR}/")
    print("  Hand the chroma_db/ folder path to Person D.")
    print("=" * 55)
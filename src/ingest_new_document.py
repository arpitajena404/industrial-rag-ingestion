import os, json, sys, subprocess

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from extract import extract_pdf, extract_word, extract_excel
from clean import clean_text, remove_boilerplate_lines
from metadata import build_document_metadata, file_checksum
from chunk import chunk_document
from embed import embed_new_chunks

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHUNKS_FILE = os.path.join(BASE_DIR, "../data/processed/chunks.jsonl")


def get_existing_checksums() -> set:
    """Reads chunks.jsonl and collects every checksum already ingested,
    so a duplicate file (even re-uploaded under a different name) is caught
    before any expensive processing happens."""
    checksums = set()
    if not os.path.exists(CHUNKS_FILE):
        return checksums
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            checksum = record.get("checksum")
            if checksum:
                checksums.add(checksum)
    return checksums


def ingest_new_document(filepath: str) -> dict:
    # Duplicate check FIRST — before extraction, embedding, or any LLM calls
    checksum = file_checksum(filepath)
    if checksum in get_existing_checksums():
        return {
            "status": "duplicate",
            "file": os.path.basename(filepath),
            "detail": "This exact file has already been ingested — skipped to avoid duplicate nodes/chunks."
        }

    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".pdf":
        raw = extract_pdf(filepath)
    elif ext in (".docx", ".doc"):
        raw = extract_word(filepath)
    elif ext in (".xlsx", ".xls"):
        raw = extract_excel(filepath)
    else:
        return {"status": "error", "detail": f"Unsupported file type: {ext}"}

    cleaned = clean_text(raw)
    if ext in (".pdf", ".docx", ".doc"):
        cleaned = remove_boilerplate_lines(cleaned)

    meta = build_document_metadata(filepath, raw)
    chunks = chunk_document(cleaned, meta)
    if not chunks:
        return {"status": "error", "detail": "No chunks produced — check the file content."}

    with open(CHUNKS_FILE, "a", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")

    embedded_count = embed_new_chunks(chunks)

    # Use sys.executable (not a bare "python") to guarantee the SAME interpreter
    # that's running this FastAPI server — avoids silently hitting a different
    # Python install that lacks neo4j/groq/python-dotenv on PATH.
    # capture_output=True + printing stdout/stderr means a real failure shows
    # up in the uvicorn terminal instead of just becoming graph_updated: false.
    try:
        result1 = subprocess.run(
            [sys.executable, os.path.join(BASE_DIR, "extract_entities.py")],
            cwd=BASE_DIR, capture_output=True, text=True
        )
        print("=== extract_entities.py stdout ===")
        print(result1.stdout)
        if result1.stderr:
            print("=== extract_entities.py stderr ===")
            print(result1.stderr)

        result2 = subprocess.run(
            [sys.executable, os.path.join(BASE_DIR, "graph_load.py")],
            cwd=BASE_DIR, capture_output=True, text=True
        )
        print("=== graph_load.py stdout ===")
        print(result2.stdout)
        if result2.stderr:
            print("=== graph_load.py stderr ===")
            print(result2.stderr)

        graph_updated = (result1.returncode == 0 and result2.returncode == 0)

    except Exception as e:
        print(f"Subprocess execution failed entirely: {e}")
        graph_updated = False

    return {
        "status": "ok",
        "file": os.path.basename(filepath),
        "doc_type": meta["doc_type"],
        "chunks_created": len(chunks),
        "chunks_embedded": embedded_count,
        "graph_updated": graph_updated,
    }
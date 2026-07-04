"""
End-to-end ingestion runner: walks the data/ folder, extracts text
(extract.py), cleans it (clean.py), attaches metadata (metadata.py), chunks
it (chunk.py), and writes every chunk as one JSON line to
data/processed/chunks.jsonl.

That file is the handoff point to the embedding stage: each line is a
self-contained record with `text` (ready for the embedding model) plus
metadata (equipment tags, doc_type, section, source file, etc.) for storing
alongside the vector and later filtering / graph linking.

Usage:
    python src/main.py
"""

import os
import sys
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from extract import extract_pdf, extract_word, extract_excel
from clean import clean_text, remove_boilerplate_lines
from metadata import build_document_metadata
from chunk import chunk_document

PDF_EXTS = {".pdf"}
WORD_EXTS = {".docx", ".doc"}
EXCEL_EXTS = {".xlsx", ".xls"}

DATA_DIR = "data"
OUTPUT_DIR = "data/processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "chunks.jsonl")


def process_file(path: str, ext: str):
    if ext in PDF_EXTS:
        raw_text = extract_pdf(path)
    elif ext in WORD_EXTS:
        raw_text = extract_word(path)
    elif ext in EXCEL_EXTS:
        raw_text = extract_excel(path)
    else:
        return []

    cleaned = clean_text(raw_text)
    if ext in PDF_EXTS or ext in WORD_EXTS:
        cleaned = remove_boilerplate_lines(cleaned)

    doc_metadata = build_document_metadata(path, raw_text)
    return chunk_document(cleaned, doc_metadata)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' does not exist.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_chunks = 0
    total_files = 0
    flagged_files = []  # low-yield extractions, likely need OCR re-processing

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for root, _, files in os.walk(DATA_DIR):
            if os.path.abspath(root).startswith(os.path.abspath(OUTPUT_DIR)):
                continue
            for file in files:
                path = os.path.join(root, file)
                ext = os.path.splitext(file.lower())[1]
                if ext not in PDF_EXTS | WORD_EXTS | EXCEL_EXTS:
                    continue

                print(f"Processing: {path}")
                try:
                    chunks = process_file(path, ext)
                except Exception as e:
                    print(f"  [FAILED] {path}: {e}")
                    continue

                total_files += 1
                if not chunks:
                    flagged_files.append((path, "0 chunks produced -- likely scanned/empty, needs OCR review"))
                    print(f"  [WARNING] No chunks produced (possible OCR needed)")
                    continue

                if chunks[0].get("needs_review"):
                    flagged_files.append((path, "low extracted text volume -- flagged for review"))

                for chunk in chunks:
                    out.write(json.dumps(chunk, ensure_ascii=False) + "\n")

                total_chunks += len(chunks)
                print(f"  -> {len(chunks)} chunks")

    print("\n" + "=" * 60)
    print(f"Done. {total_files} files processed, {total_chunks} chunks written to {OUTPUT_FILE}")
    if flagged_files:
        print(f"\n{len(flagged_files)} file(s) flagged for review:")
        for path, reason in flagged_files:
            print(f"  - {path}: {reason}")
    print("=" * 60)


if __name__ == "__main__":
    main()

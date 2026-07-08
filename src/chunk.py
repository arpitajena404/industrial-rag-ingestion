import re
import uuid


# ---------------------------------------------------------------------------
# Generic recursive text splitter (no external dependencies).
# Tries to split on the largest separator first (paragraph breaks), and only
# falls back to smaller separators (sentence, word, char) for oversized
# pieces, so chunks break at natural boundaries whenever possible.
# ---------------------------------------------------------------------------

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _split_recursive(text: str, chunk_size: int, separators: list) -> list:
    if len(text) <= chunk_size:
        return [text] if text else []

    sep = separators[0]
    remaining_seps = separators[1:]

    if sep == "":
        # Last resort: hard character split.
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    parts = text.split(sep)
    chunks = []
    current = ""

    for part in parts:
        candidate = current + (sep if current else "") + part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(part) > chunk_size:
                chunks.extend(_split_recursive(part, chunk_size, remaining_seps))
                current = ""
            else:
                current = part

    if current:
        chunks.append(current)

    return chunks


def _apply_overlap(chunks: list, overlap: int) -> list:
    if overlap <= 0 or len(chunks) <= 1:
        return chunks
    result = [chunks[0]]
    for i in range(1, len(chunks)):
        tail = chunks[i - 1][-overlap:]
        result.append(tail + " " + chunks[i])
    return result


def split_text(text: str, chunk_size: int = 900, overlap: int = 150,
               separators: list = None) -> list:
    """Recursively splits text into chunks of roughly chunk_size characters,
    preferring paragraph/sentence boundaries, with `overlap` characters of
    the previous chunk prepended to each following chunk for context
    continuity across the split point.
    """
    if not text:
        return []
    raw_chunks = _split_recursive(text, chunk_size, separators or DEFAULT_SEPARATORS)
    raw_chunks = [c.strip() for c in raw_chunks if c.strip()]
    return _apply_overlap(raw_chunks, overlap)


# ---------------------------------------------------------------------------
# Section detection for prose documents (Word/PDF).
# Heuristic: short, punctuation-free, upper-case or title-case lines that
# aren't table rows are treated as section headings (matches the real
# convention seen in work orders / SOPs, e.g. "SCOPE OF WORK", "WORK PERFORMED").
# ---------------------------------------------------------------------------

def _looks_like_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 70 or "|" in stripped:
        return False
    if stripped.endswith((".", ",", ";", ":")):
        return False
    word_count = len(stripped.split())
    if word_count > 8:
        return False
    if stripped.isupper() and word_count >= 2:
        return True
    if stripped.istitle() and word_count <= 6:
        return True
    return False


def split_into_sections(text: str) -> list:
    """Splits prose text into (heading, body) sections. Text before the first
    detected heading is returned under heading=None ('preamble')."""
    lines = text.split("\n")
    sections = []
    current_heading = None
    current_body = []

    for line in lines:
        if _looks_like_heading(line):
            if current_body:
                sections.append((current_heading, "\n".join(current_body).strip()))
            current_heading = line.strip()
            current_body = []
        else:
            current_body.append(line)

    if current_body:
        sections.append((current_heading, "\n".join(current_body).strip()))

    return [(h, b) for h, b in sections if b]


def _merge_short_chunks(chunks: list, min_chars: int = 20) -> list:
    """Merges chunks shorter than min_chars into the following chunk (or the
    previous one if it's the last chunk). Diagram-heavy technical PDFs often
    have pages that are mostly figures with sparse captions/labels (e.g. a
    lone '13' or '(FORCED WATER-LUBRICATED...)'), which the heading heuristic
    can turn into standalone near-empty chunks -- these carry no retrieval
    value on their own and just add embedding noise."""
    if not chunks:
        return chunks
    merged = []
    buffer = None
    for c in chunks:
        if buffer is not None and len(buffer["text"]) < min_chars:
            c["text"] = buffer["text"] + "\n" + c["text"]
        elif buffer is not None:
            merged.append(buffer)
        buffer = c
    if buffer is not None:
        if len(buffer["text"]) < min_chars and merged:
            merged[-1]["text"] += "\n" + buffer["text"]
        else:
            merged.append(buffer)
    return merged


def chunk_prose(text: str, doc_metadata: dict, chunk_size: int = 900,
                 overlap: int = 150) -> list:
    """Chunker for Word/PDF extracted text: splits into sections by heading,
    then size-bounds each section with the recursive splitter."""
    chunks = []
    sections = split_into_sections(text) or [(None, text)]

    for section_heading, section_text in sections:
        pieces = split_text(section_text, chunk_size=chunk_size, overlap=overlap)
        for idx, piece in enumerate(pieces):
            # Prepend the section heading to the text block for semantic indexing context
            full_text = f"Section: {section_heading}\n\n{piece}" if section_heading else piece
            chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "text": full_text,
                "section": section_heading or "General",
                "chunk_index": idx,
                **doc_metadata,
            })
    return _merge_short_chunks(chunks)


# ---------------------------------------------------------------------------
# Row-aware chunker for Excel extractions.
# extract_excel() produces text like:
#   ## Sheet: Spare Parts Inventory
#   <preamble/title lines>
#   Part No. | Description | ... | Criticality      <- header row
#   SKF-6309-2RS | Deep Groove Ball Bearing ... | CRITICAL   <- data rows
#
# We detect the header row as the start of the longest run of consecutive
# lines sharing the same pipe-delimited column count, then convert every
# data row into a self-contained "key: value; key: value" record. This is
# far more retrievable than raw pipe-delimited rows, since each chunk is a
# complete, standalone fact rather than a positional slice of a table.
# ---------------------------------------------------------------------------

def _col_count(line: str) -> int:
    return line.count("|") + 1


def _find_data_run(lines: list, min_run: int = 4):
    """Finds the longest run of consecutive lines sharing the same
    (high) column count -- this is the actual data-row block. Returns
    (run_start, run_len, col_count), or (None, None, None) if no table-like
    structure is found."""
    counts = [_col_count(l) if l.strip() else 0 for l in lines]
    n = len(lines)
    best = None  # (start, length, col_count)
    i = 0
    while i < n:
        c = counts[i]
        if c >= 3:
            j = i + 1
            while j < n and counts[j] == c:
                j += 1
            run_len = j - i
            if run_len >= min_run and (best is None or run_len > best[1]):
                best = (i, run_len, c)
            i = j
        else:
            i += 1
    return best if best else (None, None, None)


def _find_header_row(lines: list, run_start: int, col_count: int, lookback: int = 4):
    """Looks a few lines *before* the data run for the header row. Real-world
    spreadsheets sometimes split a header across several short lines due to
    merged cells (e.g. a "Vendor" super-header, then the real column names,
    then a trailing "Status | Remarks" fragment). We pick whichever preceding
    line has a column count closest to the data run's column count, rather
    than assuming the header is always the line directly above the data --
    that assumption breaks on merged-cell headers and silently misaligns
    every row. Returns None if nothing preceding looks header-like enough
    (caller then falls back to generic column labels rather than guessing)."""
    counts = [_col_count(l) if l.strip() else 0 for l in lines]
    best_idx, best_diff = None, None
    for k in range(max(0, run_start - lookback), run_start):
        if counts[k] >= 3:
            diff = abs(counts[k] - col_count)
            if best_diff is None or diff < best_diff:
                best_idx, best_diff = k, diff
    if best_idx is not None and best_diff <= 3:
        return best_idx
    return None


def _row_to_record(header_cells: list, row_cells: list) -> str:
    pairs = []
    for i in range(max(len(header_cells), len(row_cells))):
        key = header_cells[i].strip() if i < len(header_cells) else ""
        val = row_cells[i].strip() if i < len(row_cells) else ""
        if not val or not key:
            continue
        pairs.append(f"{key}: {val}")
    return "; ".join(pairs)


def chunk_excel(text: str, doc_metadata: dict) -> list:
    chunks = []
    sheet_blocks = re.split(r"(?m)^## Sheet: (.+)$", text)
    # re.split with a capturing group returns: [pre, sheet1_name, sheet1_body, sheet2_name, sheet2_body, ...]
    it = iter(sheet_blocks[1:])
    for sheet_name, body in zip(it, it):
        lines = [l for l in body.split("\n")]
        run_start, run_len, col_count = _find_data_run(lines)

        if run_start is None:
            # No detectable table structure -- fall back to prose chunking
            # for this sheet's text (e.g. a notes-only sheet).
            chunks.extend(chunk_prose(body, {**doc_metadata, "sheet_name": sheet_name.strip()}))
            continue

        # Two possible layouts:
        #  (a) header shares the data rows' column count and is simply the
        #      first line of the run (the common case, e.g. Spare Parts Inventory)
        #  (b) header has a different column count due to merged cells, and
        #      sits on a separate line before the run (e.g. the vendor list's
        #      "Vendor" / "Code | Company Name | ..." / "Status | Remarks" split)
        # We only prefer (b) when a genuinely close-matching preceding line
        # exists; otherwise we default to (a) rather than guessing with
        # generic labels, since (a) is correct in the vast majority of sheets.
        preceding_header_idx = _find_header_row(lines, run_start, col_count)

        if preceding_header_idx is not None:
            header_cells = lines[preceding_header_idx].split("|")
            header_confident = True
            preamble = "\n".join(l for l in lines[:preceding_header_idx] if l.strip())
            data_start = run_start
        elif run_len >= 2:
            header_cells = lines[run_start].split("|")
            header_confident = True
            preamble = "\n".join(l for l in lines[:run_start] if l.strip())
            data_start = run_start + 1
        else:
            # Single-row "table" with nothing usable as a header -- generic
            # labels, flagged for review rather than guessed.
            header_cells = [f"Column {k + 1}" for k in range(col_count)]
            header_confident = False
            preamble = "\n".join(l for l in lines[:run_start] if l.strip())
            data_start = run_start

        idx = 0
        j = data_start
        while j < len(lines):
            line = lines[j]
            if not line.strip():
                j += 1
                continue
            if _col_count(line) not in (col_count - 1, col_count, col_count + 1):
                break  # end of table (footer/notes with different shape)
            row_cells = line.split("|")
            record = _row_to_record(header_cells, row_cells)
            if record:
                chunks.append({
                    "chunk_id": str(uuid.uuid4()),
                    "text": f"[{sheet_name.strip()}] {record}",
                    "section": sheet_name.strip(),
                    "chunk_index": idx,
                    "sheet_name": sheet_name.strip(),
                    "sheet_preamble": preamble or None,
                    "header_confident": header_confident,
                    "needs_review": not header_confident,
                    **doc_metadata,
                })
                idx += 1
            j += 1

        # Any trailing lines after the table run (notes/footers) get their own chunk.
        trailing = "\n".join(l for l in lines[j:] if l.strip())
        if trailing:
            chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "text": f"[{sheet_name.strip()} - Notes] {trailing}",
                "section": f"{sheet_name.strip()} - Notes",
                "chunk_index": idx,
                "sheet_name": sheet_name.strip(),
                **doc_metadata,
            })

    return chunks


def chunk_document(cleaned_text: str, doc_metadata: dict) -> list:
    """Dispatches to the right chunker based on file_type in doc_metadata."""
    if not cleaned_text or not cleaned_text.strip():
        return []
    file_type = doc_metadata.get("file_type")
    if file_type == "xlsx":
        return chunk_excel(cleaned_text, doc_metadata)
    return chunk_prose(cleaned_text, doc_metadata)


if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from extract import extract_pdf, extract_word, extract_excel
    from clean import clean_text, remove_boilerplate_lines
    from metadata import build_document_metadata

    pdf_exts, word_exts, excel_exts = {".pdf"}, {".docx", ".doc"}, {".xlsx", ".xls"}

    for root, _, files in os.walk("data"):
        for file in files:
            path = os.path.join(root, file)
            ext = os.path.splitext(file.lower())[1]
            if ext not in pdf_exts | word_exts | excel_exts:
                continue

            if ext in pdf_exts:
                raw = extract_pdf(path)
            elif ext in word_exts:
                raw = extract_word(path)
            else:
                raw = extract_excel(path)

            cleaned = clean_text(raw)
            if ext in pdf_exts or ext in word_exts:
                cleaned = remove_boilerplate_lines(cleaned)

            meta = build_document_metadata(path, raw)
            chunks = chunk_document(cleaned, meta)

            print(f"\n=== {file} -> {len(chunks)} chunks (doc_type={meta['doc_type']}) ===")
            for c in chunks[:2]:
                print(f"  [{c['section']}] {c['text'][:150]!r}")

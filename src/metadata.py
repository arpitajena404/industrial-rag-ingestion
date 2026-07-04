import os
import re
import uuid
import hashlib
from datetime import datetime, timezone

# Filename-prefix conventions observed in this project's document set.
# Extend this map as your team adds new document naming conventions.
DOC_TYPE_PREFIXES = {
    "WO": "work_order",
    "NMR": "abnormality_report",   # e.g. NMR_..._DryRunning.docx
    "INS": "inspection_report",
    "SOP": "standard_operating_procedure",
    "AVL": "approved_vendor_list",
}

# Keyword-based fallback classification for PDFs/reference material that
# don't follow the WO_/SOP_/INS_ naming convention (manuals, standards, acts).
PDF_KEYWORD_TYPES = [
    (r"(^|[_\s])iom([_\s.]|$)", "installation_operation_maintenance_manual"),
    (r"hand\s*book", "reference_handbook"),
    (r"act", "regulation"),  # heuristic substring match, e.g. "factory_act..."
    (r"annexure", "annexure"),
    (r"^is[._]\d+", "technical_standard"),  # e.g. is.5120.1977.pdf
    (r"symbol", "reference_diagram_sheet"),
]

EQUIPMENT_TAG_PATTERN = re.compile(r"PUMP[-_]?\d+", re.IGNORECASE)
FILENAME_DATE_PATTERN = re.compile(r"(20\d{2})[_-]?(\d{2})[_-]?(\d{2})?")


def _normalize_equipment_tag(tag: str) -> str:
    digits = re.sub(r"\D", "", tag)
    return f"PUMP-{digits}"


def extract_equipment_tags(*texts) -> list:
    """Finds equipment tags like PUMP101, PUMP-102 in any of the given strings
    (filename and/or extracted content) and returns a deduped, normalized list.
    """
    tags = set()
    for text in texts:
        if not text:
            continue
        for match in EQUIPMENT_TAG_PATTERN.findall(text):
            tags.add(_normalize_equipment_tag(match))
    return sorted(tags)


def classify_doc_type(filename: str, ext: str) -> str:
    stem = os.path.splitext(filename)[0]
    prefix = stem.split("_")[0].upper()

    if prefix in DOC_TYPE_PREFIXES:
        return DOC_TYPE_PREFIXES[prefix]

    if ext == ".pdf":
        lower = filename.lower()
        for pattern, doc_type in PDF_KEYWORD_TYPES:
            if re.search(pattern, lower):
                return doc_type
        return "reference_document"

    if ext == ".xlsx":
        return "spreadsheet_register"

    return "general_document"


def extract_filename_date(filename: str):
    """Attempts to pull a date out of filenames like WO_2024_0618_... ->
    2024-06-18. Returns ISO date string or None if not confidently parseable.
    """
    match = FILENAME_DATE_PATTERN.search(filename)
    if not match:
        return None
    year, month, day_or_none = match.groups()
    month = month or "01"
    day = day_or_none or "01"
    try:
        return datetime(int(year), int(month), int(day)).date().isoformat()
    except ValueError:
        return None


def file_checksum(filepath: str) -> str:
    """SHA-256 of file contents, used for de-dup / change detection across
    ingestion runs so unchanged files can be skipped by teammates downstream."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()[:16]


def build_document_metadata(filepath: str, raw_text: str) -> dict:
    filename = os.path.basename(filepath)
    ext = os.path.splitext(filename)[1].lower()

    equipment_tags = extract_equipment_tags(filename, raw_text[:3000])
    char_count = len(raw_text) if raw_text else 0

    return {
        "doc_id": str(uuid.uuid5(uuid.NAMESPACE_URL, filepath)),
        "source_file": filename,
        "source_path": filepath,
        "file_type": ext.lstrip("."),
        "doc_type": classify_doc_type(filename, ext),
        "equipment_tags": equipment_tags,
        "filename_date": extract_filename_date(filename),
        "checksum": file_checksum(filepath),
        "char_count": char_count,
        # Flags low-yield extractions (likely scanned/image-only pages that
        # need OCR re-processing) so downstream teammates don't silently
        # embed near-empty or garbage chunks.
        "needs_review": char_count < 300,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }


if __name__ == "__main__":
    import glob

    for f in glob.glob("data/**/*.*", recursive=True):
        if os.path.splitext(f)[1].lower() not in (".pdf", ".docx", ".xlsx"):
            continue
        meta = build_document_metadata(f, "")  # content-based fields empty in this quick check
        print(meta["source_file"], "->", meta["doc_type"], meta["equipment_tags"])
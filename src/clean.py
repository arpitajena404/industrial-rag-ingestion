import re
import unicodedata
from collections import Counter

def clean_text(text: str) -> str:
    """Cleans and standardizes raw text:
    - Normalizes unicode characters to standard forms.
    - Replaces smart/curly quotes and dashes with standard ASCII.
    - Normalizes different bullet symbols to a standard hyphen list format.
    - Collapses multiple spaces and clean line-by-line whitespace.
    - Limits consecutive newlines to a maximum of two (double newlines).
    """
    if not text:
        return ""

    # 1. Normalize Unicode characters to standard representation (NFKC)
    text = unicodedata.normalize("NFKC", text)

    # 2. Replace smart/curly quotes, apostrophes, dashes, and bullet points
    replacements = {
        "“": '"',   # Left double smart quote
        "”": '"',   # Right double smart quote
        "‘": "'",   # Left single smart quote
        "’": "'",   # Right single smart quote
        "–": "-",   # En dash
        "—": "-",   # Em dash
        "•": "-",   # Bullet point
        "": "-",   # Wingdings bullet point (commonly from Word)
        "\xa0": " " # Non-breaking space
    }
    for original, replacement in replacements.items():
        text = text.replace(original, replacement)

    # 3. Standardize newlines (convert Windows \r\n to standard \n)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 4. Clean line-by-line whitespace and empty column dividers (|)
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        # Collapse multiple spaces/tabs into a single space, and strip outer spaces
        cleaned_line = re.sub(r"[ \t]+", " ", line).strip()
        
        # Remove trailing empty pipe columns at the end of lines (e.g. "Text | | | |" -> "Text")
        cleaned_line = re.sub(r'(?:\s*\|\s*)+$', '', cleaned_line).strip()
        
        # Compress multiple consecutive empty pipes in the middle of a line to at most 2 empty columns
        cleaned_line = re.sub(r'(?:\s*\|\s*){3,}', ' | | ', cleaned_line)
        
        cleaned_lines.append(cleaned_line)

    # 5. Rejoin and limit consecutive blank lines to at most 2 newlines (for paragraph spacing)
    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)

    return cleaned_text.strip()


def remove_boilerplate_lines(text: str, min_repeats: int = 5, max_line_len: int = 100) -> str:
    """Removes lines that repeat many times across a single document, e.g. page
    headers/footers, letterheads, fax numbers, page numbers that pdfplumber
    extracts once per physical page. Only strips short, exactly-repeating lines
    so real repeated content (e.g. table rows, 'CRITICAL', section names) is
    left untouched. Skip lines containing '|' since those are table rows
    handled separately by the excel/table chunker.

    NOTE: extract_pdf() currently concatenates all pages into one string with
    no page markers, so this works on exact line repetition rather than
    per-page position. This is a reasonable proxy since headers/footers repeat
    verbatim on every page, but if page boundaries are added to extract.py
    later, a position-aware version would be more precise.
    """
    if not text:
        return ""

    lines = text.split("\n")
    candidates = [
        l.strip() for l in lines
        if l.strip() and len(l.strip()) <= max_line_len and "|" not in l
    ]
    counts = Counter(candidates)
    boilerplate = {line for line, cnt in counts.items() if cnt >= min_repeats}

    if not boilerplate:
        return text

    kept_lines = [l for l in lines if l.strip() not in boilerplate]
    result = "\n".join(kept_lines)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


if __name__ == "__main__":
    import os
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from extract import extract_pdf, extract_word, extract_excel

    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' does not exist.")
    else:
        pdf_exts = {".pdf"}
        word_exts = {".docx", ".doc"}
        excel_exts = {".xlsx", ".xls"}
        
        # Recursively find and process all files in data directory
        for root, _, files in os.walk(data_dir):
            for file in files:
                file_path = os.path.join(root, file)
                _, ext = os.path.splitext(file.lower())
                
                # Check if file format is supported
                if ext not in pdf_exts and ext not in word_exts and ext not in excel_exts:
                    continue
                
                print(f"\n==================================================")
                print(f"Cleaning data from: {file_path}")
                print(f"==================================================")
                
                try:
                    if ext in pdf_exts:
                        raw_text = extract_pdf(file_path)
                    elif ext in word_exts:
                        raw_text = extract_word(file_path)
                    elif ext in excel_exts:
                        raw_text = extract_excel(file_path)
                        
                    cleaned_text = clean_text(raw_text)
                    if ext in pdf_exts or ext in word_exts:
                        cleaned_text = remove_boilerplate_lines(cleaned_text)
                    print(f"Raw characters: {len(raw_text)} | Cleaned characters: {len(cleaned_text)}")
                    print("\n--- Preview of CLEANED text (First 350 characters) ---")
                    print(cleaned_text[:350])
                    print("--------------------------------------------------")
                except Exception as e:
                    print(f"Failed to clean {file_path}: {e}")
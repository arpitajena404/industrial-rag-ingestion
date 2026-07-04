import re
import unicodedata

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
                    print(f"Raw characters: {len(raw_text)} | Cleaned characters: {len(cleaned_text)}")
                    print("\n--- Preview of CLEANED text (First 350 characters) ---")
                    print(cleaned_text[:350])
                    print("--------------------------------------------------")
                except Exception as e:
                    print(f"Failed to clean {file_path}: {e}")

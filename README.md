# Industrial RAG Ingestion Pipeline

A robust, modular Python pipeline designed to ingest, clean, and chunk industrial-grade documents—including PDFs, Word documents, and Excel spreadsheets—for retrieval-augmented generation (RAG) applications.

## Features
- **Multi-Format Extraction**: Parses PDFs (`PyMuPDF`), Word documents (`python-docx`), and Excel spreadsheets (`pandas`/`openpyxl`).
- **Table Preservation**: Identifies tables in Word and PDF documents and converts them to markdown tables. Converts Excel worksheets into structured markdown tables, ensuring tabular context (columns/rows) is preserved for downstream LLMs.
- **Table-Aware Chunking**: Intelligently chunks large spreadsheets/tables row-by-row, automatically replicating the column headers and sheet title in each chunk so context is never lost during retrieval.
- **Recursive Character Splitting**: Splitting logic respects paragraphs, sentences, and words to maintain semantic cohesion.
- **Robust Cleaning**: Normalizes unicode encoding, standardizes punctuation/dashes, eliminates smart quotes/ligatures, and handles carriage returns while preserving structural paragraph lines.
- **Rich Metadata Tagging**: Every chunk contains stable SHA-256 hashes, source file references, file type descriptors, coordinate bounds (for PDFs), page/sheet identifiers, and text statistics.

---

## Project Structure
```
your-repo/
├── data/
│   ├── pdf/        # PDF manuals, specifications
│   ├── word/       # SOPs, Word manuals
│   └── excel/      # Maintenance logs, asset registers
├── output/         # Destination folder for generated chunks
├── src/
│   ├── extract.py  # File parsers and table extractors
│   ├── clean.py    # Text cleaning and normalizers
│   ├── chunk.py    # Text splitting and table-aware chunking
│   └── main.py     # CLI Orchestrator and reporter
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation & Setup

1. **Clone and navigate to the project root**:
   ```bash
   cd industrial-rag-ingestion
   ```

2. **Create a virtual environment (optional but recommended)**:
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

Run the ingestion pipeline with default settings:
```bash
python src/main.py
```

### CLI Arguments

Customize the ingestion execution via flags:
```bash
python src/main.py [options]
```

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--data-dir` | `str` | `data` | Directory containing your source folders (`pdf`, `word`, `excel`). |
| `--output-dir` | `str` | `output` | Directory where output chunks are saved. |
| `--chunk-size` | `int` | `1000` | Target maximum character count for each chunk. |
| `--chunk-overlap`| `int` | `200` | Overlap character size between consecutive chunks. |

**Example:**
To ingest with a chunk size of 800 and 150 overlap, and save to a custom directory:
```bash
python src/main.py --data-dir data --output-dir my_output --chunk-size 800 --chunk-overlap 150
```

---

## Output Format

The output is saved as a single JSON file (`output/chunks.json`) containing a list of chunk objects:

```json
[
  {
    "chunk_id": "8a3d5b2cf...",
    "text": "## Sheet: Spare_Parts_Inventory_PumpHouseB\n\n| Item ID | Part Name | Manufacturer | Quantity |\n| --- | --- | --- | --- |\n| P102-S1 | Mechanical Seal | Flowserve | 4 |",
    "metadata": {
      "source_file": "data/excel/Spare_Parts_Inventory_PumpHouseB.xlsx",
      "file_type": "excel",
      "page_or_sheet": "Spare_Parts_Inventory_PumpHouseB",
      "word_count": 24,
      "char_count": 182
    }
  },
  {
    "chunk_id": "4e7c1d32a...",
    "text": "1. PURPOSE AND SCOPE\nThis document outlines the standard operating procedure (SOP) for starting up centrifugal pumps...",
    "metadata": {
      "source_file": "data/word/SOP_BSP_UTL_005_Rev2_CentrifugalPump_Operation.docx",
      "file_type": "word",
      "page_or_sheet": 1,
      "word_count": 150,
      "char_count": 912
    }
  }
]
```

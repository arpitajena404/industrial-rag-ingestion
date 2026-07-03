import os
import json
import argparse
from typing import List, Dict, Any

from extract import extract_document
from clean import clean_document
from chunk import chunk_document

def find_files(data_dir: str) -> List[str]:
    """Recursively finds all supported documents in the data directory."""
    # TODO: Implement file discovery (PDF, Word, Excel)
    return []

def main():
    parser = argparse.ArgumentParser(description="Industrial RAG Document Ingestion Pipeline")
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default="data", 
        help="Path to the directory containing raw data folders (pdf, word, excel)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="output", 
        help="Path to the output directory where chunks will be stored"
    )
    parser.add_argument(
        "--chunk-size", 
        type=int, 
        default=1000, 
        help="Max character count of each text chunk"
    )
    parser.add_argument(
        "--chunk-overlap", 
        type=int, 
        default=200, 
        help="Overlap character count between adjacent chunks"
    )
    
    args = parser.parse_args()
    
    # TODO: Implement the orchestrator pipeline:
    # 1. Discover files
    # 2. Loop files: extract -> clean -> chunk
    # 3. Aggregate all chunks
    # 4. Save to output directory
    print("Ingestion pipeline skeleton running.")

if __name__ == "__main__":
    main()

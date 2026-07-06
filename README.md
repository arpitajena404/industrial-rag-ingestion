# Industrial Knowledge Intelligence System
> ET AI Hackathon 2026 — Problem Statement 8

An AI-powered platform that ingests heterogeneous industrial documents
(PDF, DOCX, XLSX) and makes their collective knowledge queryable through
a RAG pipeline backed by a semantic vector store and a Neo4j knowledge graph.

## Architecture
Documents (PDF/DOCX/XLSX)
↓
Extraction + Cleaning + Chunking
↓
Embeddings → ChromaDB (semantic search)
Entity Extraction → Neo4j (relationship queries)
↓
RAG Chain (LangChain + Claude/GPT)
↓
Streamlit Chat Interface

## Tech Stack
| Layer | Tool |
|---|---|
| Document Ingestion | pdfplumber, python-docx, openpyxl, pytesseract |
| Chunking | LangChain RecursiveCharacterTextSplitter |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Store | ChromaDB |
| Knowledge Graph | Neo4j + py2neo |
| LLM / RAG | LangChain + Anthropic Claude |
| Frontend | Streamlit |

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run ingestion pipeline
python src/main.py

# Embed chunks into ChromaDB
python src/embed.py

# Launch the app
streamlit run src/app.py
```

## Demo Scenario
Built around a fictional **Bharat Steel Plant** (Pump House B).
Documents include maintenance work orders, inspection reports,
incident reports, SOPs, spare parts inventory, and vendor lists
— all cross-linked through the knowledge graph.

## Team
- [Name] — Ingestion Pipeline
- [Name] — Embeddings & RAG
- [Name] — Knowledge Graph
- [Name] — UI & Integration

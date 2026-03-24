# spac-filing-rag
Pipeline for automatically ingesting SPAC SEC filings (424B4), extracting structured data, and enabling RAG-based querying of SPAC prospectuses.

## Project Goal
Build a pipeline that:
1. Reads a SPAC registry
2. Resolves the best 424B4 filing for each company
3. Downloads and parses the filing
4. Extracts structured fields
5. Builds a searchable retrieval system for question answering

## Initial Scope
- One best 424B4 per company
- SEC EDGAR as the source of truth
- Excel file as initial input
- Registry file as the main backend

## Planned Project Structure
- `src/` → Python scripts
- `data/registry/` → company registry and resolved filing metadata
- `data/raw_filings/` → downloaded filing HTML
- `data/parsed_filings/` → cleaned text and extracted sections
- `notebooks/` → exploration and testing
- `config/` → settings
- `docs/` → notes and project documentation

## Build Order
1. Resolve 424B4 URLs
2. Download filings
3. Parse and clean filings
4. Extract structured fields
5. Chunk and index documents
6. Build query interface

## Version 1 includes:
- 424B4 resolver
- filing downloader
- HTML parser
- section-aware chunking
- local embeddings with FAISS
- retrieval script
- Gemini-based answer generation

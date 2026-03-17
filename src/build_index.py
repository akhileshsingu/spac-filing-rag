from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent.parent
CHUNKS_FILE = BASE_DIR / "data" / "registry" / "spac_chunks.jsonl"

INDEX_DIR = BASE_DIR / "data" / "index"
FAISS_DIR = INDEX_DIR / "faiss"
FAISS_INDEX_FILE = FAISS_DIR / "spac.index"
METADATA_FILE = FAISS_DIR / "metadata.jsonl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 128


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def batched(iterable, batch_size: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def main() -> None:
    print("STARTING BUILD_INDEX SCRIPT")
    print(f"Reading chunks from: {CHUNKS_FILE}")
    print(f"Writing index to: {FAISS_INDEX_FILE}")
    print(f"Writing metadata to: {METADATA_FILE}")

    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"Chunks file not found: {CHUNKS_FILE}")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    FAISS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    index = None
    total_vectors = 0

    with METADATA_FILE.open("w", encoding="utf-8") as meta_out:
        for batch_num, batch in enumerate(batched(iter_jsonl(CHUNKS_FILE), BATCH_SIZE), start=1):
            texts = [item["text"] for item in batch]

            embeddings = model.encode(
                texts,
                batch_size=BATCH_SIZE,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )

            if not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings)

            embeddings = embeddings.astype("float32")

            if index is None:
                dim = embeddings.shape[1]
                index = faiss.IndexFlatIP(dim)
                print(f"Initialized FAISS index with dimension: {dim}")

            index.add(embeddings)

            for item in batch:
                metadata_record = {
                    "chunk_id": item["chunk_id"],
                    "company_name": item["company_name"],
                    "ticker": item["ticker"],
                    "cik": item["cik"],
                    "accession_number": item["accession_number"],
                    "filing_date": item["filing_date"],
                    "section_title": item["section_title"],
                    "chunk_index_in_section": item["chunk_index_in_section"],
                    "total_chunks_in_section": item["total_chunks_in_section"],
                    "word_count": item["word_count"],
                    "text": item["text"],
                    "filing_url": item.get("filing_url"),
                    "local_raw_html_path": item.get("local_raw_html_path"),
                    "local_parsed_text_path": item.get("local_parsed_text_path"),
                    "local_sections_path": item.get("local_sections_path"),
                }
                meta_out.write(json.dumps(metadata_record) + "\n")

            total_vectors += len(batch)

            if batch_num % 100 == 0:
                print(f"Processed {total_vectors} chunks so far...")

    if index is None:
        raise RuntimeError("No embeddings were created. Index is empty.")

    faiss.write_index(index, str(FAISS_INDEX_FILE))

    print("\nBUILD SUMMARY")
    print(f"Total vectors indexed: {total_vectors}")
    print(f"FAISS index saved to: {FAISS_INDEX_FILE}")
    print(f"Metadata saved to: {METADATA_FILE}")


if __name__ == "__main__":
    main()
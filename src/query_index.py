from __future__ import annotations

import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent.parent
FAISS_DIR = BASE_DIR / "data" / "index" / "faiss"
INDEX_FILE = FAISS_DIR / "spac.index"
METADATA_FILE = FAISS_DIR / "metadata.jsonl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5


def load_metadata(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    print("LOADING INDEX...")

    if not INDEX_FILE.exists():
        raise FileNotFoundError(f"Missing index file: {INDEX_FILE}")
    if not METADATA_FILE.exists():
        raise FileNotFoundError(f"Missing metadata file: {METADATA_FILE}")

    index = faiss.read_index(str(INDEX_FILE))
    metadata = load_metadata(METADATA_FILE)

    print(f"Loaded FAISS index with {index.ntotal} vectors")
    print(f"Loaded {len(metadata)} metadata records")

    model = SentenceTransformer(MODEL_NAME)

    while True:
        query = input("\nEnter a query (or type 'quit'): ").strip()
        if query.lower() in {"exit", "quit"}:
            print("Exiting.")
            break
        if not query:
            continue

        query_embedding = model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        scores, indices = index.search(query_embedding, TOP_K)

        print("\nTOP RESULTS\n")
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0 or idx >= len(metadata):
                continue

            item = metadata[idx]

            print(f"Result #{rank}")
            print(f"Score: {score:.4f}")
            print(f"Company: {item.get('company_name')}")
            print(f"Ticker: {item.get('ticker')}")
            print(f"Filing Date: {item.get('filing_date')}")
            print(f"Section: {item.get('section_title')}")
            print(f"Chunk ID: {item.get('chunk_id')}")
            print(f"URL: {item.get('filing_url')}")
            print("Text Preview:")
            print(item.get("text", "")[:1000])
            print("-" * 80)


if __name__ == "__main__":
    main()
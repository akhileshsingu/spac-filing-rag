from __future__ import annotations

import json
import os
from pathlib import Path

import faiss
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
FAISS_DIR = BASE_DIR / "data" / "index" / "faiss"
INDEX_FILE = FAISS_DIR / "spac.index"
METADATA_FILE = FAISS_DIR / "metadata.jsonl"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = "gemini-3-flash-preview"
TOP_K = 5


def load_metadata(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_context(results: list[dict]) -> str:
    blocks = []
    for i, item in enumerate(results, start=1):
        block = f"""
Source {i}
Company: {item.get("company_name")}
Ticker: {item.get("ticker")}
Filing Date: {item.get("filing_date")}
Section: {item.get("section_title")}
URL: {item.get("filing_url")}
Text:
{item.get("text")}
""".strip()
        blocks.append(block)
    return "\n\n" + ("\n\n" + "=" * 80 + "\n\n").join(blocks)


def main() -> None:
    load_dotenv(dotenv_path=Path(".env"))

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY in .env")

    if not INDEX_FILE.exists():
        raise FileNotFoundError(f"Missing index file: {INDEX_FILE}")
    if not METADATA_FILE.exists():
        raise FileNotFoundError(f"Missing metadata file: {METADATA_FILE}")

    print("Loading FAISS index...")
    index = faiss.read_index(str(INDEX_FILE))
    metadata = load_metadata(METADATA_FILE)
    print(f"Loaded {index.ntotal} vectors and {len(metadata)} metadata rows")

    print("Loading embedding model...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    print("Loading Gemini client...")
    client = genai.Client(api_key=api_key)

    while True:
        query = input("\nEnter a question (or type 'exit'): ").strip()
        if query.lower() in {"exit", "quit"}:
            print("Exiting.")
            break
        if not query:
            continue

        query_embedding = embed_model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        scores, indices = index.search(query_embedding, TOP_K)

        retrieved = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(metadata):
                continue
            item = metadata[idx].copy()
            item["score"] = float(score)
            retrieved.append(item)

        context = build_context(retrieved)

        prompt = f"""
You are helping with SPAC IPO filing research.

Answer the user's question using ONLY the provided sources.
If the sources are insufficient, say so clearly.
Synthesize across sources when possible.
Cite sources inline as [Source 1], [Source 2], etc.
At the end, include a short "Sources Used" list with company, section, and URL.

User question:
{query}

Provided sources:
{context}
""".strip()

        response = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=prompt,
        )

        print("\nANSWER\n")
        print(response.text)
        print("\nRETRIEVED SOURCES\n")
        for i, item in enumerate(retrieved, start=1):
            print(f"[Source {i}] {item.get('company_name')} | {item.get('section_title')} | {item.get('filing_url')}")


if __name__ == "__main__":
    main()
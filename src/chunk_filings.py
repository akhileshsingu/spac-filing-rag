from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
PARSED_FILINGS_DIR = BASE_DIR / "data" / "parsed_filings"
OUTPUT_FILE = BASE_DIR / "data" / "registry" / "spac_chunks.jsonl"

MAX_WORDS = 300
OVERLAP_WORDS = 50


def normalize_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_words_with_overlap(
    text: str,
    max_words: int = MAX_WORDS,
    overlap_words: int = OVERLAP_WORDS,
) -> list[str]:
    words = text.split()

    if len(words) <= max_words:
        return [text.strip()] if text.strip() else []

    chunks: list[str] = []
    start = 0

    while start < len(words):
        end = min(start + max_words, len(words))
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words).strip()

        if chunk_text:
            chunks.append(chunk_text)

        if end == len(words):
            break

        start = end - overlap_words

    return chunks


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def make_chunk_record(
    metadata: dict[str, Any],
    section_title: str,
    chunk_text: str,
    chunk_index_in_section: int,
    total_chunks_in_section: int,
) -> dict[str, Any]:
    cik = str(metadata.get("cik", "")).strip()
    accession = str(metadata.get("accession_number", "")).strip()
    filing_date = metadata.get("filing_date")
    company_name = metadata.get("name")
    ticker = metadata.get("ticker")

    chunk_id = f"{cik}_{accession}_{section_title}_{chunk_index_in_section}"
    chunk_id = re.sub(r"\s+", "_", chunk_id)
    chunk_id = re.sub(r"[^A-Za-z0-9_\-]", "", chunk_id)

    return {
        "chunk_id": chunk_id,
        "company_name": company_name,
        "ticker": ticker,
        "cik": cik,
        "accession_number": accession,
        "filing_date": filing_date,
        "section_title": section_title,
        "chunk_index_in_section": chunk_index_in_section,
        "total_chunks_in_section": total_chunks_in_section,
        "text": chunk_text,
        "word_count": len(chunk_text.split()),
        "filing_url": metadata.get("filing_url"),
        "local_raw_html_path": metadata.get("local_raw_html_path"),
        "local_parsed_text_path": metadata.get("local_parsed_text_path"),
        "local_sections_path": metadata.get("local_sections_path"),
    }


def main() -> None:
    print("STARTING CHUNKING SCRIPT")
    print(f"Reading parsed filings from: {PARSED_FILINGS_DIR}")
    print(f"Writing chunk output to: {OUTPUT_FILE}")

    if not PARSED_FILINGS_DIR.exists():
        raise FileNotFoundError(f"Parsed filings directory not found: {PARSED_FILINGS_DIR}")

    sections_files = list(PARSED_FILINGS_DIR.glob("*/*/sections.json"))
    print(f"Found {len(sections_files)} sections.json files")

    all_chunks: list[dict[str, Any]] = []

    for sections_path in sections_files:
        filing_dir = sections_path.parent
        metadata_path = filing_dir / "metadata.json"

        if not metadata_path.exists():
            print(f"Skipping missing metadata: {sections_path}")
            continue

        try:
            sections = load_json(sections_path)
            metadata = load_json(metadata_path)

            for section in sections:
                section_title = normalize_text(str(section.get("section_title", "Unknown Section")))
                 # --- CLEAN SECTION TITLE ---
                section_title = section_title.strip()

                # Remove trailing punctuation
                section_title = re.sub(r"[:\.\-]+$", "", section_title)

                # Normalize casing
                section_title = section_title.title()

                # Remove garbage titles
                if section_title.lower() in {
                    "table of contents",
                    "",
                    "a",
                    "u",
                    "k",
                    "8 k",
                }:
                    continue

                # Remove very short junk
                if len(section_title) <= 2:
                    continue
                content = normalize_text(str(section.get("content", "")))

                if not content:
                    continue

                split_chunks = split_words_with_overlap(content)
                total_chunks = len(split_chunks)

                for i, chunk_text in enumerate(split_chunks, start=1):
                    record = make_chunk_record(
                        metadata=metadata,
                        section_title=section_title,
                        chunk_text=chunk_text,
                        chunk_index_in_section=i,
                        total_chunks_in_section=total_chunks,
                    )
                    all_chunks.append(record)

        except Exception as exc:
            print(f"ERROR processing {sections_path}: {exc}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for record in all_chunks:
            f.write(json.dumps(record) + "\n")

    print("\nCHUNKING SUMMARY")
    print(f"Total filings processed: {len(sections_files)}")
    print(f"Total chunks written: {len(all_chunks)}")
    if sections_files:
        avg_chunks = len(all_chunks) / len(sections_files)
        print(f"Average chunks per filing: {avg_chunks:.2f}")


if __name__ == "__main__":
    main()
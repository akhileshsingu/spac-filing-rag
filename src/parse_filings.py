from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_FILINGS_DIR = BASE_DIR / "data" / "raw_filings"
PARSED_FILINGS_DIR = BASE_DIR / "data" / "parsed_filings"


SECTION_PATTERNS = [
    "Summary",
    "Prospectus Summary",
    "Risk Factors",
    "Use of Proceeds",
    "Dividend Policy",
    "Dilution",
    "Capitalization",
    "Management",
    "Executive Compensation",
    "Principal Stockholders",
    "Certain Relationships",
    "Underwriting",
    "Description of Securities",
    "Proposed Business",
    "Business",
    "Management's Discussion and Analysis",
    "Where You Can Find Additional Information",
    "Legal Matters",
    "Experts",
    "Financial Statements",
]


def normalize_whitespace(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def clean_text(text: str) -> str:
    text = normalize_whitespace(text)

    # remove common page noise / repeated separators
    lines = [line.strip() for line in text.splitlines()]
    cleaned_lines: list[str] = []

    for line in lines:
        if not line:
            cleaned_lines.append("")
            continue

        # skip extremely short decorative lines
        if re.fullmatch(r"[_\-.=]{3,}", line):
            continue

        # skip page numbers standing alone
        if re.fullmatch(r"\d{1,4}", line):
            continue

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    # remove script/style/noscript
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # get visible-ish text
    text = soup.get_text(separator="\n")
    return clean_text(text)


def looks_like_section_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False

    if len(stripped) > 120:
        return False

    # direct match or common heading-like pattern
    for pattern in SECTION_PATTERNS:
        if stripped.lower() == pattern.lower():
            return True
        if stripped.lower().startswith(pattern.lower()):
            return True

    # mostly title case / uppercase heuristic
    words = stripped.split()
    if 1 <= len(words) <= 10:
        alpha_chars = sum(c.isalpha() for c in stripped)
        upper_chars = sum(c.isupper() for c in stripped if c.isalpha())
        if alpha_chars > 0 and upper_chars / alpha_chars > 0.7:
            return True

    return False


def split_into_sections(text: str) -> list[dict[str, Any]]:
    lines = text.splitlines()

    sections: list[dict[str, Any]] = []
    current_title = "Document Start"
    current_lines: list[str] = []

    def flush_section() -> None:
        nonlocal current_title, current_lines
        content = "\n".join(current_lines).strip()
        if content:
            sections.append(
                {
                    "section_title": current_title,
                    "content": content,
                }
            )
        current_lines = []

    for line in lines:
        if looks_like_section_heading(line):
            flush_section()
            current_title = line.strip()
        else:
            current_lines.append(line)

    flush_section()

    return sections


def load_metadata(metadata_path: Path) -> dict[str, Any]:
    if metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    return {}


def main() -> None:
    print("STARTING PARSE SCRIPT")
    print(f"Reading raw filings from: {RAW_FILINGS_DIR}")
    print(f"Writing parsed filings to: {PARSED_FILINGS_DIR}")

    if not RAW_FILINGS_DIR.exists():
        raise FileNotFoundError(f"Raw filings directory not found: {RAW_FILINGS_DIR}")

    html_files = list(RAW_FILINGS_DIR.glob("*/*/filing.html"))
    print(f"Found {len(html_files)} filing.html files")

    PARSED_FILINGS_DIR.mkdir(parents=True, exist_ok=True)

    parsed_count = 0
    skipped_count = 0
    error_count = 0

    for html_path in tqdm(html_files, desc="Parsing filings"):
        cik = html_path.parent.parent.name
        accession = html_path.parent.name

        raw_metadata_path = html_path.parent / "metadata.json"

        parsed_dir = PARSED_FILINGS_DIR / cik / accession
        parsed_dir.mkdir(parents=True, exist_ok=True)

        parsed_text_path = parsed_dir / "parsed.txt"
        sections_path = parsed_dir / "sections.json"
        parsed_metadata_path = parsed_dir / "metadata.json"

        if parsed_text_path.exists() and sections_path.exists() and parsed_metadata_path.exists():
            skipped_count += 1
            continue

        try:
            html = html_path.read_text(encoding="utf-8", errors="ignore")
            text = extract_text_from_html(html)
            sections = split_into_sections(text)

            metadata = load_metadata(raw_metadata_path)
            metadata["local_raw_html_path"] = str(html_path.relative_to(BASE_DIR))
            metadata["local_parsed_text_path"] = str(parsed_text_path.relative_to(BASE_DIR))
            metadata["local_sections_path"] = str(sections_path.relative_to(BASE_DIR))
            metadata["section_count"] = len(sections)

            parsed_text_path.write_text(text, encoding="utf-8")
            sections_path.write_text(json.dumps(sections, indent=2), encoding="utf-8")
            parsed_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

            parsed_count += 1

        except Exception as exc:
            error_count += 1
            print(f"ERROR parsing {html_path}: {exc}")

    print("\nPARSE SUMMARY")
    print(f"Parsed: {parsed_count}")
    print(f"Skipped existing: {skipped_count}")
    print(f"Errors: {error_count}")


if __name__ == "__main__":
    main()
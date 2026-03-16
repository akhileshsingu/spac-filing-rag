from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent.parent
REGISTRY_FILE = BASE_DIR / "data" / "registry" / "spac_424b4_registry.csv"
RAW_FILINGS_DIR = BASE_DIR / "data" / "raw_filings"

HEADERS = {
    "User-Agent": "Akhilesh SPAC Research akhileshsingu1@gmail.com",
    "Accept-Encoding": "gzip, deflate",
}


def safe_string(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_cik(value: Any) -> Optional[str]:
    if pd.isna(value):
        return None

    cik_str = str(value).strip()
    cik_str = cik_str.split(".")[0]
    cik_str = re.sub(r"\D", "", cik_str)

    if not cik_str:
        return None

    return cik_str.zfill(10)


def sanitize_accession(accession: Any) -> str:
    return safe_string(accession).replace("-", "")


def build_output_paths(cik: str, accession_number: str) -> tuple[Path, Path]:
    accession_clean = sanitize_accession(accession_number)
    filing_dir = RAW_FILINGS_DIR / cik / accession_clean
    html_path = filing_dir / "filing.html"
    metadata_path = filing_dir / "metadata.json"
    return html_path, metadata_path


def download_filing(url: str) -> str:
    response = requests.get(url, headers=HEADERS, timeout=60)
    response.raise_for_status()
    return response.text


def main() -> None:
    print("STARTING DOWNLOAD SCRIPT")
    print(f"Reading registry from: {REGISTRY_FILE}")

    if not REGISTRY_FILE.exists():
        raise FileNotFoundError(f"Registry file not found: {REGISTRY_FILE}")

    df = pd.read_csv(REGISTRY_FILE)
    print(f"Loaded {len(df)} registry rows")

    ready_df = df[df["Resolution_Status"] == "READY"].copy()
    print(f"Found {len(ready_df)} READY filings to consider for download")

    RAW_FILINGS_DIR.mkdir(parents=True, exist_ok=True)

    download_count = 0
    skip_count = 0
    error_count = 0

    for _, row in tqdm(ready_df.iterrows(), total=len(ready_df), desc="Downloading filings"):
        company_name = safe_string(row.get("Name"))
        ticker = safe_string(row.get("Ticker"))
        cik = normalize_cik(row.get("CIK"))
        accession_number = safe_string(row.get("Accession_Number"))
        filing_date = safe_string(row.get("Filing_Date"))
        filing_url = safe_string(row.get("Best_424B4_URL"))
        primary_document = safe_string(row.get("Primary_Document"))

        if not cik or not accession_number or not filing_url:
            error_count += 1
            print(f"Skipping invalid row: {company_name} ({ticker})")
            continue

        html_path, metadata_path = build_output_paths(cik, accession_number)

        if html_path.exists() and metadata_path.exists():
            skip_count += 1
            continue

        html_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            html_text = download_filing(filing_url)

            html_path.write_text(html_text, encoding="utf-8")

            metadata = {
                "name": company_name,
                "ticker": ticker,
                "cik": cik,
                "accession_number": accession_number,
                "filing_date": filing_date,
                "filing_url": filing_url,
                "primary_document": primary_document,
                "local_html_path": str(html_path.relative_to(BASE_DIR)),
            }

            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

            download_count += 1

        except Exception as exc:
            error_count += 1
            print(f"ERROR downloading {company_name} ({ticker}): {exc}")

        time.sleep(0.2)

    print("\nDOWNLOAD SUMMARY")
    print(f"Downloaded: {download_count}")
    print(f"Skipped existing: {skip_count}")
    print(f"Errors: {error_count}")


if __name__ == "__main__":
    main()
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Optional
import pandas as pd
import requests
from tqdm import tqdm


# =========================
# Config
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "registry" / "Akhilesh_SPAC.xlsx"
OUTPUT_FILE = BASE_DIR / "data" / "registry" / "spac_424b4_registry.csv"

SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

# Use your real email here
HEADERS = {
    "User-Agent": "Akhilesh SPAC Research akhileshsingu1@gmail.com",
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}


# =========================
# Helpers
# =========================
def normalize_cik(value: Any) -> Optional[str]:
    """Convert a CIK value into a zero-padded 10-digit string."""
    if pd.isna(value):
        return None

    cik_str = str(value).strip()

    # Remove decimal if Excel stored it like 1234567890.0
    cik_str = cik_str.split(".")[0]

    # Keep digits only
    cik_str = re.sub(r"\D", "", cik_str)

    if not cik_str:
        return None

    return cik_str.zfill(10)


def get_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Return the first matching column name from a list of candidates."""
    lower_map = {col.lower().strip(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower().strip() in lower_map:
            return lower_map[candidate.lower().strip()]
    return None


def fetch_submissions(cik: str) -> dict[str, Any]:
    """Fetch SEC submissions JSON for a company CIK."""
    url = SEC_SUBMISSIONS_URL.format(cik=cik)
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response.json()


def build_filing_url(cik: str, accession_number: str, primary_document: str) -> str:
    """
    Construct the SEC Archives filing URL.
    cik should be zero-padded, but URL path uses non-padded integer form.
    accession path removes dashes.
    """
    cik_no_zeros = str(int(cik))
    accession_no_dashes = accession_number.replace("-", "")
    return (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik_no_zeros}/{accession_no_dashes}/{primary_document}"
    )


def extract_recent_424b4_candidates(submissions_json: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract 424B4 candidates from recent filings."""
    recent = submissions_json.get("filings", {}).get("recent", {})

    forms = recent.get("form", [])
    accession_numbers = recent.get("accessionNumber", [])
    filing_dates = recent.get("filingDate", [])
    primary_documents = recent.get("primaryDocument", [])

    candidates: list[dict[str, Any]] = []

    for form, accession, filing_date, primary_doc in zip(
        forms, accession_numbers, filing_dates, primary_documents
    ):
        if form == "424B4":
            candidates.append(
                {
                    "form": form,
                    "accession_number": accession,
                    "filing_date": filing_date,
                    "primary_document": primary_doc,
                }
            )

    return candidates


def choose_best_candidate(candidates: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """
    Simple v1 rule:
    choose the earliest 424B4 filing date.

    Why earliest?
    For a SPAC-focused sheet, the earliest 424B4 is often the IPO prospectus.
    Later we can make this smarter with text-based validation.
    """
    if not candidates:
        return None

    sorted_candidates = sorted(candidates, key=lambda x: x["filing_date"])
    return sorted_candidates[0]


# =========================
# Main pipeline
# =========================
def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_excel(INPUT_FILE)

    name_col = get_col(df, ["Name", "Company", "Company Name"])
    ticker_col = get_col(df, ["Ticker", "Symbol"])
    cik_col = get_col(df, ["CIK", "Cik"])
    link_col = get_col(df, ["Link", "URL", "SEC Link", "Company Link"])

    if cik_col is None:
        raise ValueError("Could not find a CIK column in the spreadsheet.")

    output_rows: list[dict[str, Any]] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Resolving 424B4 filings"):
        company_name = row[name_col] if name_col else None
        ticker = row[ticker_col] if ticker_col else None
        original_link = row[link_col] if link_col else None
        cik = normalize_cik(row[cik_col])

        result = {
            "Name": company_name,
            "Ticker": ticker,
            "CIK": cik,
            "Original_Link": original_link,
            "Best_424B4_URL": None,
            "Filing_Date": None,
            "Accession_Number": None,
            "Primary_Document": None,
            "Resolution_Status": None,
            "Notes": None,
        }

        if not cik:
            result["Resolution_Status"] = "MISSING_CIK"
            output_rows.append(result)
            continue

        try:
            submissions_json = fetch_submissions(cik)
            candidates = extract_recent_424b4_candidates(submissions_json)

            if not candidates:
                result["Resolution_Status"] = "NOT_FOUND"
                result["Notes"] = "No 424B4 found in recent submissions."
                output_rows.append(result)
                time.sleep(0.2)
                continue

            best = choose_best_candidate(candidates)
            assert best is not None

            filing_url = build_filing_url(
                cik=cik,
                accession_number=best["accession_number"],
                primary_document=best["primary_document"],
            )

            result["Best_424B4_URL"] = filing_url
            result["Filing_Date"] = best["filing_date"]
            result["Accession_Number"] = best["accession_number"]
            result["Primary_Document"] = best["primary_document"]
            result["Resolution_Status"] = "READY"

            if len(candidates) > 1:
                result["Notes"] = f"{len(candidates)} total 424B4 candidates found; earliest selected."
            else:
                result["Notes"] = "Single 424B4 candidate found."

        except requests.HTTPError as exc:
            result["Resolution_Status"] = "HTTP_ERROR"
            result["Notes"] = str(exc)

        except Exception as exc:
            result["Resolution_Status"] = "ERROR"
            result["Notes"] = str(exc)

        output_rows.append(result)

        # Stay polite with SEC request rate
        time.sleep(0.2)

    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nDone. Output written to:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
convert_series_to_excel.py

Converts GSE53987_series_matrix.txt(.gz) from GEO into Excel format (.xlsx).

- Skips GEO metadata lines starting with "!".
- Loads expression matrix into pandas DataFrame.
- Writes to Excel for easy inspection.

Input:
    data/raw/genes/GSE53987_series_matrix.txt.gz

Output:
    data/processed/genes/GSE53987_series_matrix.xlsx
"""

import pandas as pd
from pathlib import Path
import gzip

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_GENES = PROJECT_ROOT / "data" / "raw" / "genes"
PROCESSED_GENES = PROJECT_ROOT / "data" / "processed" / "genes"

INPUT_FILE = RAW_GENES / "GSE53987_series_matrix.txt.gz"
OUTPUT_FILE = PROCESSED_GENES / "GSE53987_series_matrix.xlsx"

def main():
    PROCESSED_GENES.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

    print(f"Reading {INPUT_FILE} ...")
    # Open .gz if needed
    open_func = gzip.open if INPUT_FILE.suffix == ".gz" else open
    with open_func(INPUT_FILE, "rt", encoding="utf-8", errors="replace") as f:
        df = pd.read_csv(f, sep="\t", comment="!", index_col=0)

    print(f"Data shape: {df.shape} (rows = probes/genes, cols = samples)")
    print("Saving to Excel ...")
    df.to_excel(OUTPUT_FILE)

    print(f"✅ Saved Excel file: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

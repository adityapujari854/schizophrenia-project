#!/usr/bin/env python3
"""
scripts/preprocess_genes.py

Robust preprocess for GSE53987 series matrix.

- Reads GEO series matrix (.txt.gz)
- Numeric conversion, missing-value handling, optional log2 transform
- Variance filter (top-k) or threshold-based
- Row-wise z-score normalization (per probe/gene)
- Saves:
    data/processed/genes/GSE53987_processed_topvar.csv
    data/processed/genes/GSE53987_processed_topvar.npy
    data/processed/genes/gene_sample_qc.csv
    data/processed/genes/samples_order.csv

Usage:
    python scripts/preprocess_genes.py
    python scripts/preprocess_genes.py --topk 5000 --log2-thresh 100 --dry-run
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_GENES = PROJECT_ROOT / "data" / "raw" / "genes"
PROCESSED_GENES = PROJECT_ROOT / "data" / "processed" / "genes"

DEFAULT_INPUT = RAW_GENES / "GSE53987_series_matrix.txt.gz"
DEFAULT_TOPK = 5000

def robust_read_series_matrix(path: Path) -> pd.DataFrame:
    """
    Try reading GEO series matrix. We attempt:
     - pandas read_csv with comment='!'
     - fallback: load as text and find expression block (lines starting with 'ID' or first non-comment header)
    """
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")

    try:
        df = pd.read_csv(path, sep="\t", comment="!", index_col=0)
        if df.shape[0] > 0 and df.shape[1] > 0:
            return df
    except Exception:
        pass

    # fallback: manual parse
    with path.open("rt", encoding="utf8", errors="ignore") as fh:
        lines = [l.rstrip("\n") for l in fh if not l.startswith("!")]
    if not lines:
        raise ValueError("No non-comment lines found in series matrix.")
    # first line should be header
    header = lines[0].split("\t")
    data = [row.split("\t") for row in lines[1:]]
    df = pd.DataFrame(data, columns=header).set_index(header[0])
    return df

def main() -> None:
    p = argparse.ArgumentParser(prog="preprocess_genes.py")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--out-dir", type=Path, default=PROCESSED_GENES)
    p.add_argument("--topk", type=int, default=DEFAULT_TOPK, help="Number of top variable probes to keep")
    p.add_argument("--dropna-thresh", type=float, default=0.8,
                   help="Keep rows with at least this fraction of non-missing samples (0..1)")
    p.add_argument("--log2-thresh", type=float, default=100.0,
                   help="If max value > this, apply log2(x+1). Use 0 to disable.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    print(f"[INFO] Input: {args.input}")
    print(f"[INFO] Output dir: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = robust_read_series_matrix(args.input)
    print(f"[INFO] Raw read shape (rows=probes/gene, cols=samples): {df.shape}")

    # Try to convert everything numeric where possible; keep non-numeric columns (like gene symbol) if present
    # If the first column is 'ID' set that as index already by robust_read_series_matrix
    # We'll keep the index as probe/gene id.
    numeric_df = df.apply(pd.to_numeric, errors="coerce")

    # Drop rows with too many missing values
    min_nonmissing = int(args.dropna_thresh * numeric_df.shape[1])
    numeric_df = numeric_df.dropna(axis=0, thresh=min_nonmissing)
    print(f"[INFO] After dropna threshold: {numeric_df.shape}")

    # Impute remaining missing with row median
    numeric_df = numeric_df.apply(lambda row: row.fillna(row.median()), axis=1)

    # Decide on log2 transform by max/quantile heuristic if enabled
    if args.log2_thresh and numeric_df.max().max() > args.log2_thresh:
        print(f"[INFO] Max value {numeric_df.max().max():.2f} exceeds log2-thresh {args.log2_thresh}: applying log2(x+1)")
        numeric_df = np.log2(numeric_df + 1)

    # Remove zero-variance rows (they don't help downstream and break scaling)
    variances_all = numeric_df.var(axis=1)
    zero_var = variances_all[variances_all == 0.0].index.tolist()
    if zero_var:
        print(f"[WARN] Dropping {len(zero_var)} zero-variance rows.")
        numeric_df = numeric_df.drop(index=zero_var)

    # Variance filter: top-k most variable rows
    variances = numeric_df.var(axis=1)
    topk = min(args.topk, numeric_df.shape[0])
    top_idx = variances.sort_values(ascending=False).head(topk).index
    df_top = numeric_df.loc[top_idx]
    print(f"[INFO] Kept top {topk} variable probes: {df_top.shape}")

    # Row-wise z-score normalization (per probe across samples)
    # We scale columns after transposing to leverage scikit-learn which scales features (samples) by column
    scaler = StandardScaler(with_mean=True, with_std=True)
    # Work on float64 numpy to avoid dtype surprises
    arr = scaler.fit_transform(df_top.T).T  # rows back to probes
    df_z = pd.DataFrame(arr, index=df_top.index, columns=df_top.columns)

    print(f"[INFO] Processed shape: {df_z.shape} (rows=probes, cols=samples)")

    if args.dry_run:
        print("[INFO] Dry-run complete. No files written.")
        return

    out_csv = args.out_dir / "GSE53987_processed_topvar.csv"
    out_npy = args.out_dir / "GSE53987_processed_topvar.npy"
    qc_file = args.out_dir / "gene_sample_qc.csv"
    samples_order_file = args.out_dir / "samples_order.csv"

    # Save processed CSV and numpy array
    # Save index (probe ids) as a column named 'probe_id' for clarity
    df_z_out = df_z.copy()
    df_z_out.index.name = "probe_id"
    df_z_out.to_csv(out_csv)
    np.save(out_npy, df_z.values)
    print(f"[OK] Saved processed CSV -> {out_csv}")
    print(f"[OK] Saved NumPy array -> {out_npy}")

    # Save sample order for downstream matching
    pd.DataFrame({"sample_id": list(df_z.columns)}).to_csv(samples_order_file, index=False)
    print(f"[OK] Saved samples order -> {samples_order_file}")

    # QC per-sample
    qc = pd.DataFrame({
        "sample_id": df_z.columns,
        "mean": df_z.mean(axis=0).values,
        "std": df_z.std(axis=0).values,
        "n_missing": df_z.isna().sum(axis=0).values
    })
    qc.to_csv(qc_file, index=False)
    print(f"[OK] Saved QC -> {qc_file}")

if __name__ == "__main__":
    main()

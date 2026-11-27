#!/usr/bin/env python3
"""
scripts/gene_feature_assembly.py

Build X_genes.npy and y_genes.npy from processed gene expression.

Default behavior (from project root):
  python scripts/gene_feature_assembly.py

Inputs (defaults):
  - data/processed/genes/GSE53987_processed_topvar.csv
  - data/raw/genes/gene_sample_metadata.csv  (optional)
  - data/processed/genes/samples_order.csv    (optional; produced by preprocess_genes.py)
  - optional: probe->gene map CSV: probe_id,gene_symbol

Outputs (defaults):
  - data/processed/X_genes.npy           (samples x features)
  - data/processed/y_genes.npy           (if metadata provided)
  - data/processed/gene_samples_list.csv
  - data/raw/genes/gene_sample_metadata_template.csv  (if labels missing)

Examples:
  # quick run with defaults
  python scripts/gene_feature_assembly.py

  # specify probe->gene mapping and custom paths
  python scripts/gene_feature_assembly.py \
    --input data/processed/genes/GSE53987_processed_topvar.csv \
    --probe-map data/processed/genes/probe2gene_map.csv \
    --out-dir data/processed

Notes:
 - The script expects the processed CSV to have rows = probes (or genes) and columns = samples.
 - If probe->gene mapping is provided, probes mapping to the same gene will be aggregated by mean.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# -----------------------
# Defaults
# -----------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_GENES = PROJECT_ROOT / "data" / "processed" / "genes" / "GSE53987_processed_topvar.csv"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "raw" / "genes" / "gene_sample_metadata.csv"
DEFAULT_SAMPLES_ORDER = PROJECT_ROOT / "data" / "processed" / "genes" / "samples_order.csv"
DEFAULT_PROBE_MAP = PROJECT_ROOT / "data" / "processed" / "genes" / "probe2gene_map.csv"

DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_RAW_GENES_DIR = PROJECT_ROOT / "data" / "raw" / "genes"

OUTPUT_X = DEFAULT_OUT_DIR / "X_genes.npy"
OUTPUT_Y = DEFAULT_OUT_DIR / "y_genes.npy"
OUTPUT_SAMPLE_LIST = DEFAULT_OUT_DIR / "gene_samples_list.csv"
TEMPLATE_METADATA = DEFAULT_RAW_GENES_DIR / "gene_sample_metadata_template.csv"


# -----------------------
# Helper functions
# -----------------------
def read_processed_matrix(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Processed gene CSV not found: {path}")
    df = pd.read_csv(path, index_col=0)
    if df.shape[0] == 0 or df.shape[1] == 0:
        raise ValueError(f"Processed gene CSV appears empty or malformed: {path}")
    return df


def read_samples_order(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"samples_order file not found: {path}")
    df = pd.read_csv(path)
    if "sample_id" in df.columns:
        return df["sample_id"].astype(str).tolist()
    # fallback: first column
    return df.iloc[:, 0].astype(str).tolist()


def read_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    df = pd.read_csv(path)
    if "sample_id" not in df.columns or "label" not in df.columns:
        raise ValueError("Metadata must contain columns 'sample_id' and 'label'")
    return df


def read_probe_map(path: Path) -> pd.Series:
    """
    Read probe->gene map CSV with header containing at least 'probe_id' and 'gene_symbol'.
    Returns a Series indexed by probe_id with gene_symbol as values.
    """
    if not path.exists():
        raise FileNotFoundError(f"Probe map not found: {path}")
    pm = pd.read_csv(path)
    # heuristics for column names
    cols = [c.lower() for c in pm.columns]
    if "probe_id" in cols and "gene_symbol" in cols:
        probe_col = pm.columns[cols.index("probe_id")]
        gene_col = pm.columns[cols.index("gene_symbol")]
    else:
        # try common alternatives
        probe_col = None
        gene_col = None
        for c in pm.columns:
            lc = c.lower()
            if probe_col is None and ("probe" in lc or "id" in lc):
                probe_col = c
            if gene_col is None and ("gene" in lc or "symbol" in lc):
                gene_col = c
        if probe_col is None or gene_col is None:
            raise ValueError("Probe map must contain probe and gene columns (probe_id,gene_symbol)")
    s = pd.Series(pm[gene_col].astype(str).values, index=pm[probe_col].astype(str).values, name="gene_symbol")
    return s


def aggregate_probes_to_genes(expr_df: pd.DataFrame, probe_map: pd.Series) -> pd.DataFrame:
    """
    Given expr_df with index=probe_ids and columns=samples, aggregate probes mapped to same gene by mean.
    Probes without mapping are dropped with a warning.
    """
    # align probe_map with expr_df index
    probe_map = probe_map.reindex(expr_df.index.astype(str))
    missing_map = probe_map[probe_map.isna()]
    if not missing_map.empty:
        print(f"[WARN] {len(missing_map)} probes have no mapping and will be dropped (example up to 5): {missing_map.index.tolist()[:5]}")
    # create DataFrame mapping
    mapping = probe_map.dropna().rename_axis("probe_id").reset_index()
    mapping.columns = ["probe_id", "gene_symbol"]
    # join expression values
    expr_df_idx = expr_df.copy()
    expr_df_idx.index = expr_df_idx.index.astype(str)
    joined = expr_df_idx.loc[mapping["probe_id"]].copy()
    joined["gene_symbol"] = mapping["gene_symbol"].values
    # groupby gene_symbol and mean-aggregate
    aggregated = joined.groupby("gene_symbol").mean()
    aggregated.index.name = "gene_symbol"
    # ensure columns order preserved
    aggregated = aggregated.loc[:, expr_df.columns]
    return aggregated


def map_labels_to_binary(labels: pd.Series) -> pd.Series:
    """
    Map textual labels to numeric: 'control'->0, 'schizophrenia'->1, other: raise error or map via heuristics.
    """
    s = labels.astype(str).str.strip().str.lower()
    mapping = {"control": 0, "healthy": 0, "normal": 0, "schizophrenia": 1, "schiz": 1, "sz": 1, "patient": 1, "case": 1}
    mapped = s.map(mapping)
    if mapped.isnull().any():
        # allow user to have other label names: print examples and raise
        bad = pd.DataFrame({"orig": labels, "mapped": mapped}).loc[mapped.isnull()]
        examples = bad["orig"].unique().tolist()[:10]
        raise ValueError(f"Found labels that cannot be mapped automatically to binary (examples: {examples})."
                         " Allowed examples (control/healthy/normal -> 0; schizophrenia/sz -> 1).")
    return mapped.astype(int)


# -----------------------
# Main assembly logic
# -----------------------
def assemble(args) -> int:
    # read expression
    expr_df = read_processed_matrix(args.input)
    print(f"[INFO] Loaded processed expression: {args.input} -> shape {expr_df.shape} (rows=probes, cols=samples)")

    # optionally map probes->genes (aggregate)
    if args.probe_map:
        probe_map = read_probe_map(args.probe_map)
        print(f"[INFO] Loaded probe->gene map with {probe_map.size} entries. Aggregating probes -> genes by mean...")
        expr_df = aggregate_probes_to_genes(expr_df, probe_map)
        print(f"[INFO] After aggregation: shape {expr_df.shape} (rows=genes, cols=samples)")
    else:
        # check index name; if it looks like probe IDs, leave as-is (rows are probes)
        print("[INFO] No probe->gene map provided. Proceeding with current row identifiers (assumed probes or gene symbols).")

    # ensure sample order using samples_order.csv if exists or CLI arg
    samples = list(expr_df.columns.astype(str))
    if args.samples_order and args.samples_order.exists():
        try:
            order = read_samples_order(args.samples_order)
            # filter to those present in expression, preserve order provided
            ordered = [s for s in order if s in samples]
            if len(ordered) == 0:
                print(f"[WARN] samples_order provided but no overlap with expression columns. Using expression column order.")
            else:
                # If some expression columns missing from order, append them at end in original order
                extra = [s for s in samples if s not in ordered]
                samples = ordered + extra
                expr_df = expr_df.loc[:, samples]
                print(f"[INFO] Reordered samples using {args.samples_order} (n={len(samples)})")
        except Exception as e:
            print(f"[WARN] Failed to use samples_order file: {e} -- continuing with expression column order.")

    # final shapes
    X = expr_df.T.values  # samples x features
    feature_names = list(expr_df.index.astype(str))
    sample_list = list(expr_df.columns.astype(str))

    # prepare out dir
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # dry-run handling
    if args.dry_run:
        print(f"[DRYRUN] Would write X -> {args.out_dir / 'X_genes.npy'} shape {X.shape}")
        if args.metadata.exists():
            print(f"[DRYRUN] Would write y -> {args.out_dir / 'y_genes.npy'} (metadata found)")
        else:
            print(f"[DRYRUN] Would write template metadata -> {TEMPLATE_METADATA}")
        return 0

    # save X and sample list (samples x features)
    np.save(args.out_dir / "X_genes.npy", X)
    pd.DataFrame({"sample_id": sample_list}).to_csv(args.out_dir / "gene_samples_list.csv", index=False)
    print(f"[OK] Saved X_genes.npy -> {args.out_dir / 'X_genes.npy'} (shape: {X.shape})")
    print(f"[OK] Saved sample list -> {args.out_dir / 'gene_samples_list.csv'} (n_samples: {len(sample_list)})")

    # handle metadata -> y
    if args.metadata.exists():
        meta = read_metadata(args.metadata)
        # align metadata order to sample_list
        meta_indexed = meta.set_index("sample_id").reindex(sample_list)
        if meta_indexed["label"].isnull().any():
            missing = meta_indexed[meta_indexed["label"].isnull()].index.tolist()
            raise ValueError(f"Some samples in expression matrix are missing labels in metadata: {missing[:10]}")
        # map labels to binary
        y = map_labels_to_binary(meta_indexed["label"])
        np.save(args.out_dir / "y_genes.npy", y.values)
        print(f"[OK] Saved y_genes.npy -> {args.out_dir / 'y_genes.npy'} (shape: {y.shape})")
    else:
        # write template for user to fill in
        pd.DataFrame({"sample_id": sample_list, "label": [""] * len(sample_list)}).to_csv(TEMPLATE_METADATA, index=False)
        print(f"[WARN] No metadata provided. Wrote template file -> {TEMPLATE_METADATA}")
        print("Please fill labels ('control' or 'schizophrenia') and re-run to produce y_genes.npy")

    return 0


# -----------------------
# CLI
# -----------------------
def parse_args(argv: List[str] | None = None):
    p = argparse.ArgumentParser(prog="gene_feature_assembly.py",
                                description="Assemble X_genes.npy and y_genes.npy from processed gene matrix.")
    p.add_argument("--input", type=Path, default=DEFAULT_PROCESSED_GENES,
                   help="Processed gene CSV (rows=probes/genes, cols=samples).")
    p.add_argument("--metadata", type=Path, default=DEFAULT_METADATA,
                   help="CSV with sample_id,label (optional).")
    p.add_argument("--samples-order", type=Path, default=DEFAULT_SAMPLES_ORDER,
                   help="Optional samples_order CSV produced by preprocess_genes.py (sample_id column).")
    p.add_argument("--probe-map", type=Path, default=None,
                   help="Optional probe->gene map CSV (columns: probe_id,gene_symbol) for aggregation.")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                   help="Output directory for X_genes.npy, y_genes.npy, gene_samples_list.csv")
    p.add_argument("--dry-run", action="store_true",
                   help="Do not write outputs, just print planned actions.")
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return assemble(args)
    except Exception as e:
        print(f"[ERROR] {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
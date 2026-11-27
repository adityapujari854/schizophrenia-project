#!/usr/bin/env python3
"""
preview_gene_expression_table.py

Creates a small CSV preview combining:
 - sample IDs (from gene_samples_list.csv)
 - gene symbols (from gene_symbol_mapping.csv)
 - actual expression values (from X_genes.npy)

Useful for verifying sample ↔ gene mappings.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--gene-map", type=str, required=True)
    p.add_argument("--n-samples", type=int, default=10)
    p.add_argument("--n-genes", type=int, default=10)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    X = np.load(data_dir / "X_genes.npy")
    samples = pd.read_csv(data_dir / "gene_samples_list.csv")["sample_id"].tolist()
    gmap = pd.read_csv(args.gene_map)

    # Select top n genes that have names
    gmap_named = gmap.head(args.n_genes)
    genes = gmap_named["gene_symbol"].tolist()
    indices = gmap_named["feature_index"].tolist()

    subset = X[:args.n_samples, indices]
    df = pd.DataFrame(subset, columns=genes)
    df.insert(0, "sample_id", samples[:args.n_samples])

    out_path = data_dir / "gene_expression_preview.csv"
    df.to_csv(out_path, index=False)
    print(f"[OK] Wrote preview to: {out_path}")
    print(df.head())

if __name__ == "__main__":
    main()

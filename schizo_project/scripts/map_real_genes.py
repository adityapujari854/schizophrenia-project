#!/usr/bin/env python3
"""
map_real_genes.py

Map a list of real gene symbols (e.g. schizophrenia genes) onto the
synthetic / processed gene-feature matrix used in this project.

Modes:
 - exact  : requires a feature-names file (one gene symbol per feature column).
 - random : randomly assign provided gene symbols to feature indices.
 - index  : validate and save an existing index->symbol CSV (user-provided).

Usage (PowerShell examples):
  # Randomly map schizo genes onto synthetic features (recommended for synthetic data)
  python .\scripts\map_real_genes.py --data-dir .\data\synthetic_fullrun --gene-list .\data\schizo_genes.csv --out results\gene_symbol_mapping.csv --mode random

  # Exact match (if you have a feature names file with length == n_features)
  python .\scripts\map_real_genes.py --data-dir .\data\processed --feature-names data/processed/genes/feature_names.csv --mode exact --gene-list .\data\schizo_genes.csv --out results\gene_symbol_mapping.csv

  # Validate an index mapping file (feature_index,gene_symbol)
  python .\scripts\map_real_genes.py --data-dir .\data\synthetic_fullrun --mode index --index-file inputs/my_index_map.csv --out results/validated_index_map.csv

Outputs:
 - CSV with columns: feature_index,gene_symbol,method (random/exact/index)
 - saves to the path given with --out
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import sys

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="data/synthetic", help="Directory containing X_genes.npy (samples x features)")
    p.add_argument("--gene-list", type=str, required=True, help="CSV or TXT with gene symbols (one per line). Header allowed. If multiple columns, first column used.")
    p.add_argument("--out", type=str, required=True, help="Output CSV path to write mapping (feature_index,gene_symbol,method)")
    p.add_argument("--mode", type=str, default="random", choices=("exact","random","index"), help="Mapping mode")
    p.add_argument("--feature-names", type=str, default="", help="(exact mode) path to CSV/TXT listing feature names (one per column or one per row).")
    p.add_argument("--index-file", type=str, default="", help="(index mode) CSV with columns: feature_index,gene_symbol")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def read_gene_list(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    # try reading as csv, fallback to simple text
    try:
        df = pd.read_csv(path, header=None)
        # if single-column and header-like strings present, try read with header=0
        if df.shape[1] == 1:
            return df.iloc[:,0].astype(str).str.strip().dropna().unique().tolist()
        else:
            # many columns -> take first column
            return df.iloc[:,0].astype(str).str.strip().dropna().unique().tolist()
    except Exception:
        # fallback read lines
        with open(path, "r", encoding="utf-8") as fh:
            lines = [l.strip() for l in fh if l.strip()]
            return list(dict.fromkeys(lines))

def load_feature_names(path: Path, n_features: int):
    if not path.exists():
        raise FileNotFoundError(path)
    # attempt csv read
    try:
        df = pd.read_csv(path, header=None)
        vals = df.values.flatten().astype(str)
        vals = [v.strip() for v in vals if v is not None and str(v).strip() != ""]
        if len(vals) == n_features:
            return vals
        # try reading as a single-row (comma separated)
        if len(vals) >= n_features:
            return vals[:n_features]
        # otherwise complain
        raise ValueError(f"Feature-names file has {len(vals)} entries but expected {n_features}")
    except Exception as e:
        raise e

def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    # Find X_genes.npy
    Xp = data_dir / "X_genes.npy"
    if not Xp.exists():
        print(f"[ERROR] Expected file not found: {Xp}")
        sys.exit(1)
    X = np.load(str(Xp))
    if X.ndim != 2:
        print(f"[ERROR] Unexpected X_genes.npy shape: {X.shape}")
        sys.exit(1)
    n_features = X.shape[1]
    print(f"[INFO] Loaded X_genes.npy -> shape {X.shape} (features = {n_features})")

    gene_list = read_gene_list(Path(args.gene_list))
    if not gene_list:
        print("[ERROR] gene list is empty")
        sys.exit(1)
    print(f"[INFO] Loaded {len(gene_list)} gene symbols from: {args.gene_list}")

    mapping_rows = []

    if args.mode == "exact":
        if not args.feature_names:
            print("[ERROR] exact mode requires --feature-names pointing to a file with feature names (one per feature)")
            sys.exit(1)
        feature_names = load_feature_names(Path(args.feature_names), n_features)
        print(f"[INFO] Loaded feature names, length matches features: {len(feature_names)}")
        # attempt to match gene_list symbols to feature_names
        # build dict: feature_index -> feature_name
        matches = []
        for i, fname in enumerate(feature_names):
            sym = fname.strip()
            if sym in gene_list:
                mapping_rows.append({"feature_index": int(i), "gene_symbol": sym, "method": "exact"})
                matches.append(sym)
        print(f"[INFO] exact mode: found {len(matches)} direct matches between feature-names and provided gene list")
        # also optionally save all feature_names -> empty mapping for unmatched if user wants; but primary goal is matched ones
    elif args.mode == "index":
        if not args.index_file:
            print("[ERROR] index mode requires --index-file with columns feature_index,gene_symbol")
            sys.exit(1)
        idxp = Path(args.index_file)
        if not idxp.exists():
            print(f"[ERROR] Index file not found: {idxp}")
            sys.exit(1)
        df = pd.read_csv(idxp)
        if "feature_index" not in df.columns or "gene_symbol" not in df.columns:
            print("[ERROR] index-file must contain columns: feature_index,gene_symbol")
            sys.exit(1)
        for _, r in df.iterrows():
            fi = int(r["feature_index"])
            if fi < 0 or fi >= n_features:
                print(f"[WARN] skipping out-of-range feature_index: {fi}")
                continue
            mapping_rows.append({"feature_index": int(fi), "gene_symbol": str(r["gene_symbol"]).strip(), "method": "index"})
        print(f"[INFO] index mode: validated {len(mapping_rows)} mappings from index file")
    else:  # random
        # Randomly choose k distinct feature indices to attach provided gene symbols to,
        # if gene_list shorter than n_features we map all genes to distinct indices.
        k = min(len(gene_list), n_features)
        chosen_indices = rng.choice(n_features, size=k, replace=False)
        for i, gi in enumerate(gene_list[:k]):
            mapping_rows.append({"feature_index": int(chosen_indices[i]), "gene_symbol": gi, "method": "random"})
        print(f"[INFO] random mode: assigned {len(mapping_rows)} gene symbols to random feature indices (of {n_features} features)")

    # If no matches found (common case with synthetic), tell user and exit non-zero
    if not mapping_rows:
        print("[WARN] No mapping rows generated. If you used synthetic data, try --mode random to force mapping.")
        sys.exit(0)

    df_out = pd.DataFrame(mapping_rows).drop_duplicates(subset=["feature_index","gene_symbol"])
    # sort by feature_index for readability
    df_out = df_out.sort_values("feature_index").reset_index(drop=True)
    df_out.to_csv(out_path, index=False)
    print(f"[OK] Wrote mapping CSV to: {out_path}  (rows={len(df_out)})")
    print("[EXAMPLE] First 10 mappings:")
    print(df_out.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
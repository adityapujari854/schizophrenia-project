#!/usr/bin/env python3
"""
transform_single_sample.py

Project a single-sample feature vector into PCA space using an existing PCA cache (.npz).
Intended use: convert a large full-dimensional patient vector (genes / brain / snps)
into the much smaller PCA vector that your models expect, so you can upload that PCA
vector to a web UI or call patient_predict without re-fitting PCA.

Usage:
  python scripts/transform_single_sample.py \
      --cache results/pca_brain.npz \
      --input uploads/patient_brain_raw.npy \
      --out uploads/patient_brain_pca.npy

Also supports CSV input:
  python scripts/transform_single_sample.py \
      --cache results/pca_gene.npz \
      --input uploads/patient_gene.csv \
      --sample-index 0 \
      --out uploads/patient_gene_pca.npy

Notes:
 - The PCA cache must include PCA components and a mean. Common keys include
   'components','components_','mean','mean_'.
 - If the cache contains only 'X_pca' (full transformed matrix) then this script cannot
   project new data (we need components & mean). It will error in that case.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import sys

def find_components_and_mean(npz):
    """Return (components, mean) where components shape = (n_components, n_features)."""
    keys = set(npz.files)
    # common key names for components / mean used across scripts
    comp_keys = ["components", "components_", "pca_components", "loadings", "C", "components_array"]
    mean_keys = ["mean", "mean_", "pca_mean"]
    comp_key = next((k for k in comp_keys if k in keys), None)
    mean_key = next((k for k in mean_keys if k in keys), None)

    # allow fallback: some of our caching code stored full X_pca only (can't project)
    if comp_key is None:
        # try to infer from shapes if possible (rare)
        # but if there's 'X_pca' and nothing else, we can't project
        if "X_pca" in keys and ("components" not in keys and "mean" not in keys):
            return None, None, "cache_has_only_X_pca"
        return None, None, None

    components = npz[comp_key]
    mean = npz[mean_key] if mean_key in keys else None
    return components, mean, (comp_key, mean_key)

def read_input_vector(path: Path, sample_index: int = 0, verbose: bool = False):
    """Read a single sample vector from .npy or .csv.
    Returns 2D array shape (1, n_features)
    """
    suf = path.suffix.lower()
    if suf == ".npy":
        arr = np.load(str(path))
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        elif arr.ndim == 2:
            # if multiple rows choose sample_index
            if arr.shape[0] > 1:
                if sample_index >= arr.shape[0]:
                    raise ValueError(f"sample_index {sample_index} out of bounds for input with {arr.shape[0]} rows")
                return arr[[sample_index], :]
            return arr
        else:
            raise ValueError("Unsupported .npy array dimensions: must be 1D or 2D")
    elif suf in (".csv", ".tsv"):
        # try pandas
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError("CSV input empty")
        if sample_index >= len(df):
            raise ValueError(f"sample_index {sample_index} out of bounds for CSV with {len(df)} rows")
        row = df.iloc[sample_index].values.astype(float)
        return row.reshape(1, -1)
    else:
        raise ValueError("Unsupported input file type (expect .npy or .csv)")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True, help="PCA cache .npz file (contains components & mean)")
    p.add_argument("--input", required=True, help="Input single-sample file (.npy or .csv)")
    p.add_argument("--out", required=False, help="Output .npy filename (defaults to <input>_pca.npy)")
    p.add_argument("--csv-out", action="store_true", help="Also write CSV version of PCA vector")
    p.add_argument("--sample-index", type=int, default=0, help="Index when input contains multiple rows (CSV or 2D .npy)")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    cache_path = Path(args.cache)
    in_path = Path(args.input)
    out_path = Path(args.out) if args.out else in_path.with_name(in_path.stem + "_pca.npy")

    if args.verbose:
        print(f"[INFO] cache={cache_path}, input={in_path}, out={out_path}, sample_index={args.sample_index}")

    if not cache_path.exists():
        print(f"[ERROR] PCA cache not found: {cache_path}")
        sys.exit(1)
    if not in_path.exists():
        print(f"[ERROR] Input file not found: {in_path}")
        sys.exit(1)

    # load cache
    try:
        with np.load(str(cache_path), allow_pickle=False) as z:
            components, mean, meta = find_components_and_mean(z)
            if components is None and mean is None:
                if meta == "cache_has_only_X_pca":
                    print("[ERROR] PCA cache contains only 'X_pca' (already transformed data). "
                          "This cache cannot be used to project a new sample. Please recreate cache with components & mean.")
                    sys.exit(1)
                print(f"[ERROR] Could not find PCA components in cache. Available keys: {z.files}")
                sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to read cache {cache_path}: {e}")
        sys.exit(1)

    if args.verbose:
        print(f"[INFO] Found components shape={components.shape} mean_present={mean is not None}")

    if mean is None:
        print(f"[WARN] PCA cache has components but no mean. Proceeding with zero-mean assumption.")
        mean = np.zeros(components.shape[1], dtype=float)

    # read input vector
    try:
        X = read_input_vector(in_path, sample_index=args.sample_index, verbose=args.verbose)
    except Exception as e:
        print(f"[ERROR] Failed to read input {in_path}: {e}")
        sys.exit(1)

    if args.verbose:
        print(f"[INFO] Read input vector shape={X.shape}")

    # check dims
    n_features_required = components.shape[1]  # components: (n_components, n_features)
    if X.shape[1] != n_features_required:
        print(f"[ERROR] Feature dimension mismatch: PCA components expect {n_features_required} features, "
              f"but input has {X.shape[1]} features.")
        print(" - Possible causes: wrong input vector (e.g. brain vector from different preproc), or different ordering.")
        sys.exit(1)

    # center and project
    Xc = X - mean.reshape(1, -1)
    # components shape: (n_components, n_features) => projection = Xc dot components.T
    X_pca = np.dot(Xc, components.T)

    # save output
    try:
        np.save(str(out_path), X_pca)
        if args.csv_out:
            csv_out = out_path.with_suffix(".csv")
            import pandas as pd
            pd.DataFrame(X_pca).to_csv(csv_out, index=False, header=False)
            if args.verbose:
                print(f"[INFO] Wrote CSV PCA vector: {csv_out}")
        if args.verbose:
            print(f"[OK] Wrote PCA vector: {out_path} (shape={X_pca.shape})")
        else:
            print(out_path)
    except Exception as e:
        print(f"[ERROR] Failed to write output {out_path}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

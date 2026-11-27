#!/usr/bin/env python3
"""
reduce_snps_to_pca.py

Fit PCA on X_snps.npy and save a cache (npz) and metadata JSON.

Usage:
    python scripts/reduce_snps_to_pca.py --data-dir data/synthetic_fullrun --out-dir results --n-components 50 --save-full --verbose
"""
from pathlib import Path
import argparse
import numpy as np
from sklearn.decomposition import PCA
import json
import time
import os

def safe_save_npz_atomic(path: Path, **kw):
    tmp = path.with_suffix(path.suffix + f".tmp{int(time.time()*1000)}")
    np.savez(tmp, **kw)
    try:
        os.replace(str(tmp), str(path))
    except Exception:
        tmp.rename(path)
    return path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="data/synthetic", help="Directory with X_snps.npy")
    p.add_argument("--out-dir", type=str, default="results", help="Where to write pca cache")
    p.add_argument("--n-components", type=int, default=50, help="PCA components")
    p.add_argument("--save-full", action="store_true", help="Include full X_pca in the npz (large)")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_path = data_dir / "X_snps.npy"
    if not x_path.exists():
        print(f"[ERROR] X_snps.npy not found at: {x_path}")
        raise SystemExit(1)

    if args.verbose:
        print(f"[INFO] Loading X_snps from: {x_path}")
    X = np.load(str(x_path))
    n, p = X.shape
    if args.verbose:
        print(f"[INFO] X_snps shape: {X.shape}")

    n_comp = min(args.n_components, p)
    n_comp = max(1, n_comp)

    if args.verbose:
        print(f"[INFO] Fitting PCA (n_components={n_comp}) on X_snps (shape=({n},{p})) ...")
    t0 = time.time()
    pca = PCA(n_components=n_comp, random_state=42)
    X_pca = pca.fit_transform(X)
    elapsed = time.time() - t0
    if args.verbose:
        print(f"[INFO] PCA done in {elapsed:.2f}s; X_pca shape: {X_pca.shape}")

    cache_path = out_dir / "pca_snp.npz"
    cache_dict = {
        "components": pca.components_,
        "mean": pca.mean_,
        "explained_variance_ratio": pca.explained_variance_ratio_,
    }
    if args.save_full:
        cache_dict["X_pca"] = X_pca

    try:
        safe_save_npz_atomic(cache_path, **cache_dict)
        print(f"[OK] Wrote PCA cache to: {cache_path}")
    except Exception as e:
        print(f"[WARN] Failed to write PCA cache to {cache_path}: {e}")

    meta = {
        "n_samples": int(n),
        "n_features": int(p),
        "n_components": int(n_comp),
        "fitted_at": time.time(),
    }
    meta_path = out_dir / "pca_snp_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[OK] Wrote metadata to {meta_path}")
    print("[DONE] PCA cache created.")

if __name__ == "__main__":
    main()

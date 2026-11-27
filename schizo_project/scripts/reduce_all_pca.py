#!/usr/bin/env python3
"""
reduce_all_pca.py

Build / validate PCA caches for genes, brain, and snps.

This version fixes tmp-file save issues on Windows by creating a tmp filename
that np.savez will actually write and then atomically replacing the final file.

Example:
  python scripts/reduce_all_pca.py --data-dir data/synthetic_fullrun --out-dir results \
    --n-gene 200 --n-brain 500 --n-snp 50 --save-full --verbose
"""
from pathlib import Path
import argparse
import numpy as np
import time
import os
import traceback
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from datetime import datetime

def log(msg):
    print(f"[INFO] {msg}")

def warn(msg):
    print(f"[WARN] {msg}")

def err(msg):
    print(f"[ERROR] {msg}")

def safe_save_npz(path: Path, **arrays):
    """
    Save arrays to an .npz file robustly.

    Behavior:
      - Create a tmp filename in the same directory that ends with '.npz' (so np.savez
        will write the expected file).
      - Save to the tmp file, then atomic-replace (os.replace) to final path.
      - If replace fails (permission/lock), try tmp.rename(path).
      - If that still fails due to permissions, write a timestamped alternate file and return it.
    Returns the Path actually written, or None on total failure.
    """
    out_dir = path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = int(time.time() * 1000)
    # Build a tmp filename that already ends with .npz so np.savez writes exactly that file.
    tmp_name = f"{path.stem}.tmp{ts}.npz"
    tmp_path = out_dir / tmp_name

    try:
        # np.savez will write exactly the filename we pass (it won't append extra suffix if we include .npz)
        np.savez(tmp_path, **arrays)

        # Try atomic replace
        try:
            os.replace(str(tmp_path), str(path))
            return path
        except Exception as e_replace:
            # If replace failed, try rename() as a fallback
            try:
                tmp_path.rename(path)
                return path
            except Exception as e_rename:
                warn(f"os.replace failed ({e_replace}); tmp_path.rename failed ({e_rename}). Will try alternate filename.")
                # fall through to alternate save
    except PermissionError as e:
        warn(f"PermissionError when writing tmp cache {tmp_path}: {e}")
    except Exception as e:
        warn(f"Failed to write tmp cache {tmp_path}: {e}")
        traceback.print_exc()

    # If we reached here, tmp->final swap failed. Attempt to write timestamped alternate file.
    alt_ts = datetime.now().strftime("%Y%m%d%H%M%S%f")
    alt_name = f"{path.stem}_alt_{alt_ts}.npz"
    alt_path = out_dir / alt_name
    try:
        np.savez(alt_path, **arrays)
        log(f"Wrote alternate PCA cache to: {alt_path}")
        return alt_path
    except Exception as e_alt:
        err(f"Failed to write alternate PCA cache {alt_path}: {e_alt}")
        traceback.print_exc()
        return None

def validate_cache(path: Path):
    """Return (ok, msg). Checks file exists, readable, contains X_pca and non-empty shape."""
    if not path.exists():
        return False, "missing"
    try:
        with np.load(str(path)) as npz:
            if "X_pca" not in npz:
                return False, "missing key X_pca"
            X = npz["X_pca"]
            if X.size == 0:
                return False, "empty X_pca"
            return True, f"ok shape={X.shape}"
    except Exception as e:
        return False, f"read error: {e}"

def fit_and_save_pca(X, n_components, out_path: Path, verbose=False):
    """
    Fit PCA on X and save cache. Returns dict with metadata and result path or None on failure.
    """
    n_samples, n_features = X.shape
    if n_components is None or n_components <= 0:
        n_components = min(n_samples, n_features)
    else:
        n_components = min(n_components, n_samples, n_features)

    if n_components <= 0:
        raise ValueError("n_components must be > 0 after bounds checking")

    if verbose:
        log(f"Fitting PCA (n_components={n_components}) on X shape={X.shape} ...")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = PCA(n_components=n_components, svd_solver="auto", random_state=42)
    t0 = time.time()
    X_pca = pca.fit_transform(Xs)
    elapsed = time.time() - t0
    if verbose:
        log(f"PCA done in {elapsed:.2f}s; X_pca shape: {X_pca.shape}")

    save_dict = {
        "X_pca": X_pca,
        "components": pca.components_,
        "mean": pca.mean_,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "fitted_at": np.array([time.time()]),
        "n_samples": np.array([int(n_samples)]),
        "n_features": np.array([int(n_features)]),
        "n_components": np.array([int(n_components)])
    }

    saved = safe_save_npz(out_path, **save_dict)
    if saved:
        ok, msg = validate_cache(saved)
        if ok:
            log(f"[OK] Wrote PCA cache to: {saved}")
            return {"path": saved, "n_components": n_components, "time_s": elapsed}
        else:
            warn(f"Saved cache but validation says: {msg}")
            return {"path": saved, "n_components": n_components, "time_s": elapsed}
    else:
        warn(f"Failed to save PCA cache to {out_path}")
        return None

def load_array(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return np.load(str(path))

def main():
    p = argparse.ArgumentParser(description="Build PCA caches for genes, brain, snps")
    p.add_argument("--data-dir", type=str, default="data/synthetic", help="Base dir with X_genes.npy etc.")
    p.add_argument("--out-dir", type=str, default="results", help="Output dir for pca_*.npz")
    p.add_argument("--n-gene", type=int, default=200, help="n_components for gene PCA")
    p.add_argument("--n-brain", type=int, default=500, help="n_components for brain PCA")
    p.add_argument("--n-snp", type=int, default=50, help="n_components for snp PCA")
    p.add_argument("--force", action="store_true", help="Force rebuild even if cache looks valid")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    Xg_path = data_dir / "X_genes.npy"
    Xb_path = data_dir / "X_brain.npy"
    Xs_path = data_dir / "X_snps.npy"

    missing = [p for p in (Xg_path, Xb_path, Xs_path) if not p.exists()]
    if missing:
        err(f"Missing input arrays: {missing}. Aborting.")
        return 1

    tasks = [
        ("gene", Xg_path, out_dir / "pca_gene.npz", args.n_gene),
        ("brain", Xb_path, out_dir / "pca_brain.npz", args.n_brain),
        ("snp", Xs_path, out_dir / "pca_snp.npz", args.n_snp),
    ]

    summary = []
    for tag, in_path, out_path, n_comp in tasks:
        print("-" * 70)
        log(f"Processing {tag}: input={in_path}, cache={out_path}, n_components={n_comp}")
        ok, msg = validate_cache(out_path)
        if ok and not args.force:
            log(f"Existing cache OK: {out_path} ({msg}) — skipping (use --force to rebuild).")
            summary.append((tag, "skipped", out_path, msg))
            continue

        if ok and args.force:
            log(f"Force rebuilding cache for {tag} (existing cache was: {msg}).")

        try:
            X = load_array(in_path)
        except Exception as e:
            err(f"Failed to load {in_path}: {e}")
            traceback.print_exc()
            summary.append((tag, "load_failed", None, str(e)))
            continue

        try:
            res = fit_and_save_pca(X, n_comp, out_path, verbose=args.verbose)
            if res is None:
                summary.append((tag, "save_failed", out_path, "save error"))
            else:
                summary.append((tag, "saved", res["path"], f"n_components={res['n_components']} time_s={res['time_s']:.2f}"))
        except Exception as e:
            err(f"PCA failed for {tag}: {e}")
            traceback.print_exc()
            summary.append((tag, "failed", out_path, str(e)))

    print("\n" + "="*70)
    log("Summary:")
    for tag, status, path, info in summary:
        print(f" - {tag:6s} : {status:8s} : {path} : {info}")
    print("="*70)
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())

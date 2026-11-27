#!/usr/bin/env python3
"""
reduce_brain_to_pca.py

Fit PCA on X_brain.npy and cache results to results/pca_brain.npz.

Usage:
  python .\scripts\reduce_brain_to_pca.py --data-dir data/synthetic_signal_targeted --n-components 500
  python .\scripts\reduce_brain_to_pca.py --data-dir data/synthetic_signal_targeted --n-components 500 --out-dir results --save-full --force

Notes:
 - By default this will save components, mean, explained_variance_ratio and (optionally) X_pca.
 - On Windows, writing is done via a temporary file and then moved to the final path to reduce race conditions.
"""

from pathlib import Path
import argparse
import numpy as np
import json
import sys
import tempfile
import shutil
import time

from sklearn.decomposition import PCA

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="data/synthetic", help="Directory containing X_brain.npy")
    p.add_argument("--out-dir", type=str, default="results", help="Directory to write pca cache (results/pca_brain.npz)")
    p.add_argument("--n-components", type=int, default=500, help="Requested number of PCA components")
    p.add_argument("--save-full", action="store_true", help="Save the transformed X_pca matrix inside the cache (can be large)")
    p.add_argument("--force", action="store_true", help="Force re-fit and overwrite existing cache")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()

def vprint(msg, verbose):
    if verbose:
        print(msg)
    else:
        print(msg)  # keep info messages visible by default

def atomic_save_npz(dst: Path, arr_dict: dict, verbose: bool = False):
    """
    Save arr_dict to dst (npz) atomically by writing to a temp file then replacing.
    Returns True on success, False on failure.
    """
    tmp_dir = dst.parent
    tmp_fd, tmp_path = tempfile.mkstemp(prefix=dst.name + ".tmp", dir=str(tmp_dir))
    try:
        # close fd, numpy will write by path
        try:
            import os
            os.close(tmp_fd)
        except Exception:
            pass
        np.savez(tmp_path, **arr_dict)
        # On Windows, replace the target file if it exists
        try:
            if dst.exists():
                dst.unlink()
        except Exception as e:
            vprint(f"[WARN] Could not remove existing cache file before renaming: {e}", verbose)
        try:
            shutil.move(tmp_path, str(dst))
        except Exception as e:
            vprint(f"[WARN] shutil.move failed: {e}. Attempting os.replace...", verbose)
            try:
                import os
                os.replace(tmp_path, str(dst))
            except Exception as e2:
                vprint(f"[ERROR] Failed to atomically write cache file: {e2}", verbose)
                # cleanup tmp
                try:
                    Path(tmp_path).unlink()
                except Exception:
                    pass
                return False
        vprint(f"[OK] Wrote PCA cache to: {dst}", verbose)
        return True
    except Exception as e:
        vprint(f"[ERROR] Failed to save PCA cache: {e}", verbose)
        try:
            Path(tmp_path).unlink()
        except Exception:
            pass
        return False

def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pca_cache = out_dir / "pca_brain.npz"
    x_path = data_dir / "X_brain.npy"
    if not x_path.exists():
        print(f"[ERROR] Missing input file: {x_path}")
        sys.exit(2)

    vprint(f"[INFO] Loading X_brain from: {x_path}", args.verbose)
    try:
        X = np.load(str(x_path))
    except Exception as e:
        print(f"[ERROR] Failed to load {x_path}: {e}")
        sys.exit(3)

    if X.ndim != 2:
        print(f"[ERROR] X_brain.npy must be 2D (n_samples, n_features). Found shape={X.shape}")
        sys.exit(4)

    n_samples, n_features = X.shape
    vprint(f"[INFO] X_brain shape: {X.shape}", args.verbose)

    # compute allowed n_components
    max_nc = min(n_samples, n_features)
    if args.n_components <= 0:
        n_components = max_nc
    else:
        n_components = min(args.n_components, max_nc)
    if n_components != args.n_components:
        vprint(f"[WARN] Requested n_components={args.n_components} is > min(n_samples, n_features)={max_nc}. Clamping to {n_components}.", True)

    # If cache exists and not forcing, try to validate it first
    if pca_cache.exists() and not args.force:
        vprint(f"[INFO] Found existing PCA cache: {pca_cache}. Validating...", args.verbose)
        try:
            npz = np.load(str(pca_cache))
            # require essential keys
            keys = set(npz.files)
            required_keys = {"components", "mean", "explained_variance_ratio"}
            if not required_keys.issubset(keys):
                vprint(f"[WARN] PCA cache missing required keys {required_keys - keys}. Will recompute.", True)
                npz.close()
            else:
                # validate shapes
                comps = npz["components"]
                mean = npz["mean"]
                evr = npz["explained_variance_ratio"]
                # basic shape checks
                if comps.ndim != 2 or mean.ndim != 1 or evr.ndim != 1:
                    vprint(f"[WARN] PCA cache arrays have unexpected shapes. Will recompute.", True)
                    npz.close()
                else:
                    # check component count
                    cached_nc = comps.shape[0]
                    if cached_nc != n_components:
                        vprint(f"[WARN] Cached n_components={cached_nc} differs from requested {n_components}. Recomputing.", True)
                        npz.close()
                    else:
                        vprint(f"[OK] PCA cache appears valid. Nothing to do.", True)
                        # still write summary
                        print(f"[OK] PCA cache is valid at: {pca_cache}")
                        sys.exit(0)
        except Exception as e:
            vprint(f"[WARN] Failed to read/validate existing cache: {e}. Will recompute.", True)

    # Fit PCA
    print(f"[INFO] Fitting PCA (n_components={n_components}) on X_brain (shape={X.shape}). This may take a while...")
    t0 = time.time()
    try:
        pca = PCA(n_components=n_components, random_state=args.random_state, svd_solver="randomized")
        X_pca = pca.fit_transform(X)
    except Exception as e:
        # fallback to 'full' solver if randomized fails
        vprint(f"[WARN] PCA fit with 'randomized' failed: {e}. Retrying with svd_solver='full'...", True)
        try:
            pca = PCA(n_components=n_components, random_state=args.random_state, svd_solver="full")
            X_pca = pca.fit_transform(X)
        except Exception as e2:
            print(f"[ERROR] PCA failed: {e2}")
            sys.exit(5)

    elapsed = time.time() - t0
    vprint(f"[INFO] PCA done in {elapsed:.2f}s; X_pca shape: {X_pca.shape}", True)

    # Prepare cache dictionary
    arr_dict = {
        "components": pca.components_,
        "mean": pca.mean_,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "n_samples": np.array([int(n_samples)]),
        "n_features": np.array([int(n_features)]),
        "n_components": np.array([int(n_components)]),
    }

    # Optionally include the transformed matrix (can be large)
    if args.save_full:
        vprint("[INFO] Including full transformed X_pca in cache (this may be large).", True)
        arr_dict["X_pca"] = X_pca

    # Attempt atomic save
    success = atomic_save_npz(pca_cache, arr_dict, verbose=args.verbose)
    if not success:
        print("[WARN] Failed to atomically save PCA cache. Attempting direct save (may fail on Windows).")
        try:
            np.savez(str(pca_cache), **arr_dict)
            print(f"[OK] Saved PCA cache to: {pca_cache} (direct save)")
        except Exception as e:
            print(f"[ERROR] Could not save PCA cache: {e}")
            # but we still have X_pca in memory and can write X_pca separately if requested
            if args.save_full:
                fallback = out_dir / "X_brain_pca.npy"
                try:
                    np.save(fallback, X_pca)
                    print(f"[OK] Wrote fallback full X_pca to: {fallback}")
                except Exception as e2:
                    print(f"[ERROR] Could not save fallback X_pca either: {e2}")
            sys.exit(6)

    # Also save a small JSON metadata for human reading
    meta = {
        "created_at": time.time(),
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "n_components": int(n_components),
        "save_full_matrix": bool(args.save_full),
        "elapsed_s": float(elapsed)
    }
    try:
        with open(out_dir / "pca_brain_meta.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        vprint(f"[OK] Wrote metadata to {out_dir / 'pca_brain_meta.json'}", args.verbose)
    except Exception as e:
        vprint(f"[WARN] Could not write PCA metadata JSON: {e}", args.verbose)

    print("[DONE] PCA cache created.")
    sys.exit(0)

if __name__ == "__main__":
    main()

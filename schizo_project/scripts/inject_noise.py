#!/usr/bin/env python3
"""
inject_noise.py

Inject synthetic noise into existing (synthetic) datasets to test robustness.

Two modes:
 - --copy-all : copy base dataset to out-dir and overwrite canonical filenames (X_genes.npy, y_genes.npy, etc.)
 - default    : create *_noisy.npy files in out-dir (leaves base files unchanged)

Noises supported:
 - random label flips (mislabel some fraction of samples)
 - missing data: set some entries to NaN
 - feature drop: zero out or remove fraction of features (here we zero-out)
 - gaussian additive noise (per-feature SD scaled)
 - swap labels among a small fraction (alternative mislabel behavior)

Usage examples (PowerShell):
  # produce noisy copy with 5% mislabeled, 2% missing entries, small gaussian noise
  python .\scripts\inject_noise.py --data-dir data\synthetic_signal_strongest --out-dir data\synthetic_signal_strongest_noisy --copy-all --overwrite --mislabel-fraction 0.05 --missing-fraction 0.02 --gaussian-noise-sd 0.1

  # create *_noisy files only (non-destructive)
  python .\scripts\inject_noise.py --data-dir data\synthetic --out-dir data\synthetic_noisy --mislabel-fraction 0.01 --feature-drop-fraction 0.01

"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import shutil
import json
import sys

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="data/synthetic", help="Source data dir (contains X_*.npy, y_*.npy)")
    p.add_argument("--out-dir", type=str, default="data/synthetic_noisy", help="Output directory for noisy dataset")
    p.add_argument("--mislabel-fraction", type=float, default=0.0, help="Fraction (0-1) of samples to flip labels (randomly)")
    p.add_argument("--swap-fraction", type=float, default=0.0, help="Fraction (0-1) of samples to swap labels pairwise (alternative)")
    p.add_argument("--missing-fraction", type=float, default=0.0, help="Fraction (0-1) of matrix entries to set to NaN")
    p.add_argument("--feature-drop-fraction", type=float, default=0.0, help="Fraction (0-1) of features to zero-out (per-modality)")
    p.add_argument("--gaussian-noise-sd", type=float, default=0.0, help="Add Gaussian noise (sd fraction of per-feature sd). E.g. 0.1 -> 0.1 * feature_sd")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--copy-all", action="store_true", help="Copy base dataset into out-dir and overwrite canonical names (recommended)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing files in out-dir without prompt")
    return p.parse_args()

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _load_np(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return np.load(str(path))

def copy_base_files(src_dir: Path, dst_dir: Path, overwrite: bool = False):
    patterns = ["*.npy", "*.csv", "*.json"]
    for pat in patterns:
        for f in src_dir.glob(pat):
            dst = dst_dir / f.name
            if dst.exists() and not overwrite:
                continue
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)

def inject_noise_to_matrix(X, rng, missing_fraction=0.0, feature_drop_fraction=0.0, gauss_sd_fraction=0.0):
    """Return modified copy of X with the requested noises injected."""
    X2 = X.copy().astype(float)  # ensure float for NaN
    n, p = X2.shape

    # feature drop: choose features to zero-out
    if feature_drop_fraction and feature_drop_fraction > 0:
        n_drop = int(round(feature_drop_fraction * p))
        if n_drop > 0:
            drop_idx = rng.choice(p, size=min(n_drop, p), replace=False)
            X2[:, drop_idx] = 0.0
    # gaussian additive noise (per-feature sd scaling)
    if gauss_sd_fraction and gauss_sd_fraction > 0:
        # compute per-feature sd on original X (ddof=0)
        feature_sd = np.std(X, axis=0, ddof=0)
        # build noise matrix
        noise = rng.normal(loc=0.0, scale=1.0, size=X2.shape) * (feature_sd * gauss_sd_fraction)
        # replace NaNs in feature_sd with 1.0 to avoid 0 scale
        noise = np.nan_to_num(noise)
        X2 += noise
    # missing entries: randomly set some entries to NaN
    if missing_fraction and missing_fraction > 0:
        total = X2.size
        n_missing = int(round(total * missing_fraction))
        if n_missing > 0:
            # pick flat indices and set to NaN
            flat_idx = rng.choice(total, size=n_missing, replace=False)
            rows = (flat_idx // p)
            cols = (flat_idx % p)
            X2[rows, cols] = np.nan
    return X2

def mislabel_or_swap(y, rng, mislabel_fraction=0.0, swap_fraction=0.0):
    """Return modified labels and log of changed indices."""
    y2 = y.copy()
    n = y2.shape[0]
    changed = {"flipped": [], "swapped_pairs": []}

    # simple flips: change label to 1-label
    if mislabel_fraction and mislabel_fraction > 0:
        n_flip = int(round(mislabel_fraction * n))
        if n_flip > 0:
            idx = rng.choice(n, size=min(n_flip, n), replace=False)
            y2[idx] = 1 - y2[idx]
            changed["flipped"] = idx.tolist()

    # swap pairwise: pick 2*k samples and swap labels pairwise among them
    if swap_fraction and swap_fraction > 0:
        n_swap = int(round(swap_fraction * n))
        if n_swap > 1:
            if n_swap % 2 == 1:
                n_swap -= 1
            idx = rng.choice(n, size=min(n_swap, n), replace=False)
            # pair them and swap labels
            for i in range(0, len(idx), 2):
                a, b = idx[i], idx[i+1]
                tmp = y2[a].copy()
                y2[a] = y2[b]
                y2[b] = tmp
                changed["swapped_pairs"].append([int(a), int(b)])
    return y2, changed

def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)
    src = Path(args.data_dir)
    out = Path(args.out_dir)
    safe_mkdir(out)

    # canonical filenames we expect in base synthetic datasets
    files_required = {
        "genes": ("X_genes.npy", "y_genes.npy", "gene_samples_list.csv"),
        "brain": ("X_brain.npy", "y_brain.npy", "brain_subjects_list.csv"),
        "snps": ("X_snps.npy", "y_snps.npy", "snp_samples_list.csv"),
        "meta": ("metadata.csv",)
    }

    # Check base files exist in source
    missing = []
    for m in ("genes", "brain", "snps"):
        for fn in files_required[m][:2]:
            if not (src / fn).exists():
                missing.append(str(src / fn))
    if missing:
        print("[ERROR] The following required files are missing from data-dir:")
        for mm in missing:
            print("  -", mm)
        sys.exit(1)

    # If copy-all: copy base files into out dir first
    if args.copy_all:
        print(f"[INFO] Copying base dataset from {src} -> {out} (overwrite={args.overwrite})")
        copy_base_files(src, out, overwrite=args.overwrite)

    # load canonical arrays from source (always start from original)
    Xg = _load_np(src / "X_genes.npy")
    Xb = _load_np(src / "X_brain.npy")
    Xs = _load_np(src / "X_snps.npy")
    # labels: prefer y_genes or y_signal / y if present (we assume y is shared across modalities)
    y_candidates = ["y_genes.npy", "y_signal.npy", "y.npy"]
    y = None
    for yc in y_candidates:
        p = src / yc
        if p.exists():
            y = _load_np(p); break
    if y is None:
        y = _load_np(src / "y_genes.npy")
    # sample lists / metadata
    gene_list = pd.read_csv(src / "gene_samples_list.csv") if (src / "gene_samples_list.csv").exists() else None
    brain_list = pd.read_csv(src / "brain_subjects_list.csv") if (src / "brain_subjects_list.csv").exists() else None
    snp_list = pd.read_csv(src / "snp_samples_list.csv") if (src / "snp_samples_list.csv").exists() else None
    meta = pd.read_csv(src / "metadata.csv") if (src / "metadata.csv").exists() else None

    print(f"[INFO] Loaded shapes: Xg={Xg.shape}, Xb={Xb.shape}, Xs={Xs.shape}, y={y.shape}")

    # --- inject noises per modality ---
    Xg2 = inject_noise_to_matrix(Xg, rng,
                                missing_fraction=args.missing_fraction,
                                feature_drop_fraction=args.feature_drop_fraction,
                                gauss_sd_fraction=args.gaussian_noise_sd)
    Xb2 = inject_noise_to_matrix(Xb, rng,
                                missing_fraction=args.missing_fraction,
                                feature_drop_fraction=args.feature_drop_fraction,
                                gauss_sd_fraction=args.gaussian_noise_sd)
    Xs2 = inject_noise_to_matrix(Xs, rng,
                                missing_fraction=args.missing_fraction,
                                feature_drop_fraction=args.feature_drop_fraction,
                                gauss_sd_fraction=args.gaussian_noise_sd)

    # --- label corruptions ---
    y2, label_changes = mislabel_or_swap(y, rng,
                                        mislabel_fraction=args.mislabel_fraction,
                                        swap_fraction=args.swap_fraction)

    # Save: either overwrite canonical names in out-dir (if copy-all) or save *_noisy files
    if args.copy_all:
        print("[INFO] Writing modified canonical arrays into out-dir (overwrite canonical filenames).")
        np.save(out / "X_genes.npy", Xg2)
        np.save(out / "X_brain.npy", Xb2)
        np.save(out / "X_snps.npy", Xs2)
        np.save(out / "y_genes.npy", y2)
        # copy or write sample lists & metadata if missing in out
        if gene_list is not None:
            (out / "gene_samples_list.csv").write_text(gene_list.to_csv(index=False))
        if brain_list is not None:
            (out / "brain_subjects_list.csv").write_text(brain_list.to_csv(index=False))
        if snp_list is not None:
            (out / "snp_samples_list.csv").write_text(snp_list.to_csv(index=False))
        if meta is not None:
            (out / "metadata.csv").write_text(meta.to_csv(index=False))
        save_mode = "canonical"
    else:
        # non-destructive: save with _noisy suffix
        print("[INFO] Writing noisy arrays with '_noisy' suffix (base files left untouched).")
        np.save(out / "X_genes_noisy.npy", Xg2)
        np.save(out / "X_brain_noisy.npy", Xb2)
        np.save(out / "X_snps_noisy.npy", Xs2)
        np.save(out / "y_noisy.npy", y2)
        if gene_list is not None:
            gene_list.to_csv(out / "gene_samples_list.csv", index=False)
        if brain_list is not None:
            brain_list.to_csv(out / "brain_subjects_list.csv", index=False)
        if snp_list is not None:
            snp_list.to_csv(out / "snp_samples_list.csv", index=False)
        if meta is not None:
            meta.to_csv(out / "metadata.csv", index=False)
        save_mode = "suffixed"

    # Save a small summary JSON describing injected noise
    summary = {
        "source_dir": str(src),
        "out_dir": str(out),
        "save_mode": save_mode,
        "n_samples": int(y.shape[0]),
        "mislabel_fraction": float(args.mislabel_fraction),
        "swap_fraction": float(args.swap_fraction),
        "missing_fraction": float(args.missing_fraction),
        "feature_drop_fraction": float(args.feature_drop_fraction),
        "gaussian_noise_sd": float(args.gaussian_noise_sd),
        "label_changes": label_changes,
        "seed": int(args.seed)
    }
    with open(out / "noise_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"[OK] Wrote noisy dataset to: {out}")
    print("Summary:", json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()

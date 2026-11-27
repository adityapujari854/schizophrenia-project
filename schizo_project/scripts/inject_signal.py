#!/usr/bin/env python3
"""
inject_signal.py (improved)

Inject deterministic signal into gene/brain/SNP arrays to produce a dataset
with controllable predictive performance.

Supports:
 - mode=random   : choose features at random per modality
 - mode=targeted : pick features from provided gene/brain mapping CSVs

Mapping CSV format (optional):
 - Required columns for gene-map/brain-map: feature_index (int), gene_symbol (optional), method (optional)
 - You can generate mapping with `map_real_genes.py` (random mapping or attempted match)

Example usage:
  # targeted using gene map, random brain selection
  python .\scripts\inject_signal.py --data-dir data/synthetic --out-dir data/synthetic_signal_targeted \
      --mode targeted --gene-map .\results\gene_symbol_mapping.csv --n-gene-signal 150 --n-brain-signal 40 \
      --effect 1.4 --fraction 0.8 --copy-all --overwrite

  # random selection (no copy)
  python .\scripts\inject_signal.py --data-dir data/synthetic --out-dir data/signal_rand --no-copy \
      --n-gene-signal 100 --n-brain-signal 20 --effect 0.8 --fraction 0.6

"""
from pathlib import Path
import numpy as np
import pandas as pd
import argparse
import shutil
import json
import sys
import os
import time

def parse_args():
    p = argparse.ArgumentParser(description="Inject signal into gene/brain/snp arrays")
    p.add_argument("--data-dir", type=str, default="data/synthetic", help="Input data directory")
    p.add_argument("--out-dir", type=str, default="data/synthetic_signal", help="Output dir for signal dataset")
    p.add_argument("--mode", choices=("random", "targeted"), default="random", help="How to choose signal features")
    p.add_argument("--gene-map", type=str, default=None, help="CSV with gene -> feature_index mapping (for targeted mode)")
    p.add_argument("--brain-map", type=str, default=None, help="CSV with brain -> feature_index mapping (for targeted mode)")
    p.add_argument("--n-gene-signal", type=int, default=50, help="Number of gene features to carry signal")
    p.add_argument("--n-brain-signal", type=int, default=20, help="Number of brain features to carry signal")
    p.add_argument("--n-snp-signal", type=int, default=0, help="Number of SNP features to carry signal")
    p.add_argument("--effect", type=float, default=0.8, help="Effect size in SD units added to positive class")
    p.add_argument("--fraction", type=float, default=0.8, help="Fraction of positive class samples to modify (0-1)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--copy-all", dest="copy_all", action="store_true", help="Copy whole base dataset then overwrite canonical files (default)")
    group.add_argument("--no-copy", dest="copy_all", action="store_false", help="Do not copy base files; write *_signal.npy files instead")
    p.set_defaults(copy_all=True)
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing files in out-dir")
    return p.parse_args()

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def copy_base_files(src_dir: Path, dst_dir: Path, overwrite: bool = False):
    patterns = ["*.npy", "*.csv"]
    for pat in patterns:
        for f in sorted(src_dir.glob(pat)):
            dst = dst_dir / f.name
            if dst.exists() and not overwrite:
                # skip copying existing file
                continue
            shutil.copy2(f, dst)

def _load_np(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return np.load(str(path))

def load_map_csv(path: Path):
    """Load a mapping CSV and return sorted unique feature indices."""
    df = pd.read_csv(path)
    # Prefer column names 'feature_index' or 'feature' or 'index'
    candidate_cols = [c for c in df.columns if c.lower() in ("feature_index","feature","index","feature_idx","feat_idx")]
    if not candidate_cols:
        raise ValueError(f"No column named feature_index/feature/index found in mapping CSV: {path}")
    fi_col = candidate_cols[0]
    indices = pd.to_numeric(df[fi_col], errors="coerce").dropna().astype(int).unique().tolist()
    return sorted(indices), df

def pick_targeted_indices(available_indices, requested_n, mapped_indices, rng):
    """
    available_indices: full-range [0..n_features-1]
    requested_n: how many to pick
    mapped_indices: list of indices coming from map (subset of available_indices)
    """
    mapped = [i for i in mapped_indices if i in available_indices]
    if not mapped:
        # fallback to random
        return rng.choice(available_indices, size=min(requested_n, len(available_indices)), replace=False).tolist(), False
    # if mapped has fewer than requested, pick all mapped and then fill randomly
    if len(mapped) >= requested_n:
        sel = rng.choice(mapped, size=requested_n, replace=False).tolist()
        return sel, True
    else:
        # use all mapped, then fill remaining from available (excluding already chosen)
        chosen = set(mapped)
        remaining_need = requested_n - len(mapped)
        pool = [i for i in available_indices if i not in chosen]
        fill = rng.choice(pool, size=min(remaining_need, len(pool)), replace=False).tolist() if pool else []
        return sorted(list(chosen) + fill), True

def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    safe_mkdir(out_dir)

    # required base files in input data_dir
    required = [
        "X_genes.npy", "X_brain.npy", "X_snps.npy", "y_genes.npy",
        "gene_samples_list.csv", "brain_subjects_list.csv", "snp_samples_list.csv", "metadata.csv"
    ]
    missing = [f for f in required if not (data_dir / f).exists()]
    if missing:
        print("ERROR: Missing required base files in data-dir:", missing)
        sys.exit(1)

    # if copy_all: copy base files into out_dir first
    if args.copy_all:
        print(f"[INFO] Copying base dataset from {data_dir} -> {out_dir} (overwrite={args.overwrite})")
        try:
            copy_base_files(data_dir, out_dir, overwrite=args.overwrite)
        except Exception as e:
            print(f"[ERROR] Failed copying files: {e}")
            sys.exit(1)

    # load arrays from original data_dir (always base from original)
    print("[INFO] Loading arrays from:", data_dir)
    Xg = _load_np(data_dir / "X_genes.npy")
    Xb = _load_np(data_dir / "X_brain.npy")
    Xs = _load_np(data_dir / "X_snps.npy")
    y = _load_np(data_dir / "y_genes.npy")
    gene_list = pd.read_csv(data_dir / "gene_samples_list.csv")
    brain_list = pd.read_csv(data_dir / "brain_subjects_list.csv")
    snp_list = pd.read_csv(data_dir / "snp_samples_list.csv")
    meta = pd.read_csv(data_dir / "metadata.csv")

    n, ng = Xg.shape
    _, nb = Xb.shape
    _, ns = Xs.shape

    if not (len(y) == n == len(gene_list)):
        print("ERROR: sample count mismatch between X_genes, y_genes, and gene_samples_list.csv")
        sys.exit(1)

    # If mode targeted, try to read maps
    gene_map_indices = []
    brain_map_indices = []
    gene_map_df = None
    brain_map_df = None
    if args.mode == "targeted":
        if args.gene_map:
            gm_path = Path(args.gene_map)
            if not gm_path.exists():
                print(f"[WARN] Provided gene-map not found: {gm_path} -- will fallback to random selection for genes")
            else:
                try:
                    gene_map_indices, gene_map_df = load_map_csv(gm_path)
                    print(f"[INFO] Loaded gene-map: {len(gene_map_indices)} unique mapped indices")
                except Exception as e:
                    print(f"[WARN] Failed to load gene-map CSV: {e}; falling back to random for genes")
                    gene_map_indices = []
        if args.brain_map:
            bm_path = Path(args.brain_map)
            if not bm_path.exists():
                print(f"[WARN] Provided brain-map not found: {bm_path} -- will fallback to random selection for brain features")
            else:
                try:
                    brain_map_indices, brain_map_df = load_map_csv(bm_path)
                    print(f"[INFO] Loaded brain-map: {len(brain_map_indices)} unique mapped indices")
                except Exception as e:
                    print(f"[WARN] Failed to load brain-map CSV: {e}; falling back to random for brain features")
                    brain_map_indices = []

    # choose indices for injection
    # genes
    all_gene_indices = list(range(ng))
    if args.mode == "random":
        gene_idx = rng.choice(all_gene_indices, size=min(args.n_gene_signal, ng), replace=False) if args.n_gene_signal>0 else np.array([],dtype=int)
        gene_idx = np.array(sorted(gene_idx))
        gene_map_used = False
    else:
        sel, used = pick_targeted_indices(all_gene_indices, args.n_gene_signal, gene_map_indices, rng)
        gene_idx = np.array(sorted(sel))
        gene_map_used = used
        if not gene_map_indices:
            print("[WARN] mode=targeted but no gene-map provided or loaded; selected gene indices randomly as fallback.")

    # brain
    all_brain_indices = list(range(nb))
    if args.mode == "random":
        brain_idx = rng.choice(all_brain_indices, size=min(args.n_brain_signal, nb), replace=False) if args.n_brain_signal>0 else np.array([],dtype=int)
        brain_idx = np.array(sorted(brain_idx))
        brain_map_used = False
    else:
        selb, usedb = pick_targeted_indices(all_brain_indices, args.n_brain_signal, brain_map_indices, rng)
        brain_idx = np.array(sorted(selb))
        brain_map_used = usedb
        if not brain_map_indices:
            print("[WARN] mode=targeted but no brain-map provided or loaded; selected brain indices randomly as fallback.")

    # snps (we don't have targeted mapping generally)
    all_snp_indices = list(range(ns))
    if args.n_snp_signal > 0:
        snp_idx = rng.choice(all_snp_indices, size=min(args.n_snp_signal, ns), replace=False)
        snp_idx = np.array(sorted(snp_idx))
    else:
        snp_idx = np.array([], dtype=int)

    # choose positive-class samples to receive effect
    pos_idx = np.where(y == 1)[0]
    if len(pos_idx) == 0:
        print("ERROR: No positive-class (label==1) samples found in y_genes.npy")
        sys.exit(1)
    n_pos = len(pos_idx)
    n_signal_pos = int(round(max(0.0, min(1.0, args.fraction)) * n_pos))
    signal_pos_idx = rng.choice(pos_idx, size=n_signal_pos, replace=False) if n_signal_pos > 0 else np.array([], dtype=int)

    # compute per-feature SD (ddof=0) to create effect as effect * sd
    gene_sd = np.std(Xg, axis=0, ddof=0)
    brain_sd = np.std(Xb, axis=0, ddof=0)
    snp_sd = np.std(Xs, axis=0, ddof=0)

    # copy arrays
    Xg2 = Xg.copy()
    Xb2 = Xb.copy()
    Xs2 = Xs.copy()

    # Add mean shift for selected features (positive-class samples)
    for fi in gene_idx:
        shift = args.effect * (gene_sd[fi] if gene_sd[fi] > 0 else 1.0)
        Xg2[signal_pos_idx, fi] += shift
    for fi in brain_idx:
        shift = args.effect * (brain_sd[fi] if brain_sd[fi] > 0 else 1.0)
        Xb2[signal_pos_idx, fi] += shift
    for fi in snp_idx:
        shift = args.effect * (snp_sd[fi] if snp_sd[fi] > 0 else 1.0)
        Xs2[signal_pos_idx, fi] += shift

    # Save outputs
    def _save_overwrite(dst: Path, arr):
        if dst.exists() and not args.overwrite:
            print(f"[SKIP] {dst} exists (use --overwrite to replace).")
        else:
            np.save(dst, arr)

    if args.copy_all:
        print("[INFO] Writing modified arrays into out-dir using canonical names (overwrite={})".format(args.overwrite))
        safe_mkdir(out_dir)
        try:
            _save_overwrite(out_dir / "X_genes.npy", Xg2)
            _save_overwrite(out_dir / "X_brain.npy", Xb2)
            _save_overwrite(out_dir / "X_snps.npy", Xs2)
            _save_overwrite(out_dir / "y_genes.npy", y)
        except PermissionError as e:
            print(f"[ERROR] Permission error saving files to {out_dir}: {e}")
            print("Try closing any program holding these files (Excel/Editor) or use a different out-dir.")
            sys.exit(1)
        # also copy / overwrite lists & metadata
        for df_name, df in [("gene_samples_list.csv", gene_list), ("brain_subjects_list.csv", brain_list),
                            ("snp_samples_list.csv", snp_list), ("metadata.csv", meta)]:
            dst = out_dir / df_name
            if dst.exists() and not args.overwrite:
                print(f"[SKIP] {dst.name} exists (use --overwrite to replace).")
            else:
                df.to_csv(dst, index=False)
    else:
        safe_mkdir(out_dir)
        np.save(out_dir / "X_genes_signal.npy", Xg2)
        np.save(out_dir / "X_brain_signal.npy", Xb2)
        np.save(out_dir / "X_snps_signal.npy", Xs2)
        np.save(out_dir / "y_signal.npy", y)
        gene_list.to_csv(out_dir / "gene_samples_list.csv", index=False)
        brain_list.to_csv(out_dir / "brain_subjects_list.csv", index=False)
        snp_list.to_csv(out_dir / "snp_samples_list.csv", index=False)
        meta.to_csv(out_dir / "metadata.csv", index=False)

    # Save summary describing what was injected
    summary = {
        "n_samples": int(n),
        "n_gene_signal": int(gene_idx.size),
        "n_brain_signal": int(brain_idx.size),
        "n_snp_signal": int(snp_idx.size),
        "effect": float(args.effect),
        "fraction_positive_with_signal": float(args.fraction),
        "n_positive": int(n_pos),
        "n_positive_with_signal": int(n_signal_pos),
        "signal_samples_example": signal_pos_idx[:20].tolist(),
        "gene_signal_examples": gene_idx[:50].tolist(),
        "brain_signal_examples": brain_idx[:50].tolist(),
        "snp_signal_examples": snp_idx[:50].tolist(),
        "mode": args.mode,
        "gene_map_used": bool(gene_map_used),
        "brain_map_used": bool(brain_map_used)
    }
    with open(out_dir / "signal_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"Wrote signal dataset to: {out_dir}")
    print("Signal gene indices (examples):", summary["gene_signal_examples"][:10])
    print("Signal brain indices (examples):", summary["brain_signal_examples"][:10])
    print("Positive samples with signal (examples):", summary["signal_samples_example"][:10])
    print("Done.")

if __name__ == "__main__":
    main()

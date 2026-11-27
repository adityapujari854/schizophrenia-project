#!/usr/bin/env python3
"""
generate_synthetic_dataset.py

Create a canonical synthetic dataset for the project.

Outputs (saved to --out-dir):
  X_genes.npy            (n_samples x n_gene_features)  float32
  y_genes.npy            (n_samples,)                   int8 (0/1)
  gene_samples_list.csv  (sample_id)
  X_brain.npy            (n_samples x n_brain_features) float32
  y_brain.npy            (n_samples,)                   int8 (copy of y_genes)
  brain_subjects_list.csv
  brain_qc.csv           (simple QC columns)
  X_snps.npy             (n_samples x n_snp_features)   float32 (dosage-like)
  y_snps.npy             (n_samples,)                   int8 (copy of y_genes)
  snp_samples_list.csv
  metadata.csv           (sample-level metadata)
  signal_summary.json    (empty/default - placeholder)

Usage (PowerShell):
  python .\scripts\generate_synthetic_dataset.py --out-dir .\data\synthetic --n-samples 10000
  python .\scripts\generate_synthetic_dataset.py --out-dir .\data\synthetic --n-samples 2000 --quick

This script is deterministic given --seed.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import json
import sys

def parse_args():
    p = argparse.ArgumentParser(description="Generate canonical synthetic dataset")
    p.add_argument("--out-dir", type=str, default="data/synthetic", help="Output directory for synthetic dataset")
    p.add_argument("--n-samples", type=int, default=10000, help="Number of synthetic samples to generate")
    p.add_argument("--n-gene-features", type=int, default=5000, help="Number of gene features (columns)")
    p.add_argument("--n-brain-features", type=int, default=19900, help="Number of brain connectivity features (columns)")
    p.add_argument("--n-snp-features", type=int, default=1000, help="Number of SNP features (columns)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--quick", action="store_true", help="Quick mode: generate much smaller dataset for testing")
    return p.parse_args()

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def make_ids(prefix: str, n: int) -> list:
    width = max(4, len(str(n)))
    return [f"{prefix}{str(i+1).zfill(width)}" for i in range(n)]

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    safe_mkdir(out_dir)

    # Quick override for fast dev
    if args.quick:
        n = min(args.n_samples, 2000)
        ng = min(args.n_gene_features, 500)
        nb = min(args.n_brain_features, 2000)
        ns = min(args.n_snp_features, 200)
        print(f"[INFO] QUICK mode: n_samples={n}, gene={ng}, brain={nb}, snp={ns}")
    else:
        n = args.n_samples
        ng = args.n_gene_features
        nb = args.n_brain_features
        ns = args.n_snp_features

    rng = np.random.RandomState(args.seed)

    # Sample IDs
    gene_sample_ids = make_ids("sample-", n)
    brain_sample_ids = make_ids("sub-", n)   # brain uses sub- prefix in many pipelines
    snp_sample_ids = make_ids("snp-sample-", n)

    print(f"[INFO] Generating gene matrix: n={n}, features={ng} (float32)")
    # Genes: standard normal, some small per-feature variance differences
    gene_base = rng.normal(loc=0.0, scale=1.0, size=(n, ng)).astype(np.float32)
    # add tiny per-feature scaling to create feature variance heterogeneity
    gene_scales = 0.5 + rng.rand(ng).astype(np.float32)  # 0.5-1.5
    gene_base *= gene_scales[np.newaxis, :]

    print(f"[INFO] Generating brain matrix: n={n}, features={nb} (float32)")
    # Brain: smaller variance features
    brain_base = rng.normal(loc=0.0, scale=0.05, size=(n, nb)).astype(np.float32)
    brain_scales = 0.5 + 0.2 * rng.rand(nb).astype(np.float32)  # small heterogeneity
    brain_base *= brain_scales[np.newaxis, :]

    print(f"[INFO] Generating SNP matrix (dosage-like): n={n}, features={ns} (float32)")
    # SNPs: dosage approx in [0,2], modeled as beta -> scaled to 0-2 around population AF
    af = 0.3 + 0.4 * rng.rand(ns).astype(np.float32)  # allele freq 0.3-0.7
    # sample from binomial(2, af) with slight noise and float
    snp_base = rng.binomial(2, af, size=(n, ns)).astype(np.float32)
    # add tiny gaussian noise to make them non-integer
    snp_base += rng.normal(0, 0.02, size=(n, ns)).astype(np.float32)

    # Labels: balanced-ish binary labels
    # create half 0 / half 1 (if odd, last sample labelled 0)
    n_pos = n // 2
    n_neg = n - n_pos
    y = np.array([1] * n_pos + [0] * n_neg, dtype=np.int8)
    # shuffle samples consistently across modalities
    perm = rng.permutation(n)
    gene_base = gene_base[perm]
    brain_base = brain_base[perm]
    snp_base = snp_base[perm]
    y = y[perm]
    # create sample id lists in the shuffled order
    gene_sample_ids_shuf = [gene_sample_ids[i] for i in perm]
    brain_sample_ids_shuf = [brain_sample_ids[i] for i in perm]
    snp_sample_ids_shuf = [snp_sample_ids[i] for i in perm]

    # Save arrays with canonical names expected by pipeline
    print("[INFO] Saving arrays and CSVs to:", out_dir)
    np.save(out_dir / "X_genes.npy", gene_base)
    np.save(out_dir / "y_genes.npy", y)
    pd.DataFrame({"sample_id": gene_sample_ids_shuf}).to_csv(out_dir / "gene_samples_list.csv", index=False)

    np.save(out_dir / "X_brain.npy", brain_base)
    np.save(out_dir / "y_brain.npy", y)  # same labels
    pd.DataFrame({"subject_id": brain_sample_ids_shuf}).to_csv(out_dir / "brain_subjects_list.csv", index=False)

    # simple brain QC file (fake values but plausible)
    brain_qc = pd.DataFrame({
        "subject_id": brain_sample_ids_shuf,
        "n_vols": rng.randint(120, 240, size=n),
        "mean_FD": np.abs(rng.normal(0.05, 0.03, size=n).astype(np.float32)),
        "pct_FD_gt_0.5": np.clip(rng.normal(0.01, 0.02, size=n), 0.0, 1.0)
    })
    brain_qc.to_csv(out_dir / "brain_qc.csv", index=False)

    np.save(out_dir / "X_snps.npy", snp_base)
    np.save(out_dir / "y_snps.npy", y)
    pd.DataFrame({"sample_id": snp_sample_ids_shuf}).to_csv(out_dir / "snp_samples_list.csv", index=False)

    # metadata.csv: minimal useful columns
    meta_df = pd.DataFrame({
        "sample_id": gene_sample_ids_shuf,
        "label": y.astype(int),
        "age": rng.randint(18, 75, size=n),
        "sex": rng.choice(["M", "F"], size=n),
        "cohort": rng.choice(["synthetic_cohort_A"], size=n)
    })
    meta_df.to_csv(out_dir / "metadata.csv", index=False)

    # placeholder summary (useful to track later)
    summary = {
        "n_samples": int(n),
        "n_gene_features": int(ng),
        "n_brain_features": int(nb),
        "n_snp_features": int(ns),
        "seed": int(args.seed),
        "note": "Canonical synthetic dataset generated by generate_synthetic_dataset.py"
    }
    with open(out_dir / "signal_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    # quick stats print
    print("[DONE] Synthetic dataset created.")
    print(f"  X_genes.npy: {gene_base.shape}  (dtype={gene_base.dtype})")
    print(f"  X_brain.npy: {brain_base.shape}  (dtype={brain_base.dtype})")
    print(f"  X_snps.npy:  {snp_base.shape}   (dtype={snp_base.dtype})")
    print(f"  y arrays:    {y.shape}          (labels: {np.bincount(y)})")
    print(f"  Files written to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()

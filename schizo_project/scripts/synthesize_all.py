#!/usr/bin/env python3
"""
synthesize_all.py

Generate a full synthetic dataset (10k subjects) for schizophrenia project.

Outputs:
- data/synthetic/metadata.csv
- data/synthetic/X_genes.npy, y_genes.npy, gene_samples_list.csv
- data/synthetic/X_brain.npy, y_brain.npy, brain_subjects_list.csv, brain_qc.csv
- data/synthetic/X_snps.npy, y_snps.npy, snp_samples_list.csv

Resume capable: skips steps if output files already exist.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ---------- CONFIG ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "data" / "synthetic"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N = 10000   # number of subjects
N_GENES = 5000
N_BRAIN_PARCELS = 200
N_SNPS = 1000
LATENT = 30
RNG = np.random.default_rng(42)


# ---------- HELPERS ----------
def already_done(path: Path):
    return path.exists()


def save_metadata():
    out = OUT_DIR / "metadata.csv"
    if already_done(out):
        print("[SKIP] metadata.csv already exists.")
        return pd.read_csv(out)

    print("[MAKE] metadata.csv")
    ids = [f"sub-{i:05d}" for i in range(1, N+1)]
    labels = RNG.choice(["control", "schizophrenia"], size=N)
    ages = RNG.integers(18, 65, size=N)
    sexes = RNG.choice(["M", "F"], size=N)
    sites = RNG.choice(["site1", "site2", "site3"], size=N)

    df = pd.DataFrame({
        "sample_id": ids,
        "age": ages,
        "sex": sexes,
        "site": sites,
        "label": labels
    })
    df.to_csv(out, index=False)
    print("[OK] metadata.csv saved")
    return df


def save_genes(meta):
    X_path = OUT_DIR / "X_genes.npy"
    y_path = OUT_DIR / "y_genes.npy"
    list_path = OUT_DIR / "gene_samples_list.csv"

    if all(already_done(p) for p in [X_path, y_path, list_path]):
        print("[SKIP] Gene dataset already exists.")
        return

    print("[MAKE] Gene dataset")
    n = len(meta)
    Z = RNG.normal(size=(n, LATENT))
    W = RNG.normal(size=(LATENT, N_GENES))
    X = Z @ W + RNG.normal(scale=0.5, size=(n, N_GENES))
    y = (meta["label"].str.lower() == "schizophrenia").astype(int).values

    np.save(X_path, X)
    np.save(y_path, y)
    pd.DataFrame({"sample_id": meta["sample_id"]}).to_csv(list_path, index=False)

    print(f"[OK] Gene dataset saved (X={X.shape}, y={y.shape})")


def save_brain(meta):
    X_path = OUT_DIR / "X_brain.npy"
    y_path = OUT_DIR / "y_brain.npy"
    list_path = OUT_DIR / "brain_subjects_list.csv"
    qc_path = OUT_DIR / "brain_qc.csv"

    if all(already_done(p) for p in [X_path, y_path, list_path, qc_path]):
        print("[SKIP] Brain dataset already exists.")
        return

    print("[MAKE] Brain dataset")
    n = len(meta)
    Z = RNG.normal(size=(n, LATENT))
    W = RNG.normal(size=(LATENT, N_BRAIN_PARCELS * (N_BRAIN_PARCELS - 1) // 2))
    X = Z @ W + RNG.normal(scale=0.1, size=(n, W.shape[1]))
    y = (meta["label"].str.lower() == "schizophrenia").astype(int).values

    np.save(X_path, X)
    np.save(y_path, y)
    pd.DataFrame({"sample_id": meta["sample_id"]}).to_csv(list_path, index=False)

    motion = RNG.uniform(0, 0.5, size=n)
    pd.DataFrame({"sample_id": meta["sample_id"], "fd_mean": motion}).to_csv(qc_path, index=False)

    print(f"[OK] Brain dataset saved (X={X.shape}, y={y.shape})")


def save_snps(meta):
    X_path = OUT_DIR / "X_snps.npy"
    y_path = OUT_DIR / "y_snps.npy"
    list_path = OUT_DIR / "snp_samples_list.csv"

    if all(already_done(p) for p in [X_path, y_path, list_path]):
        print("[SKIP] SNP dataset already exists.")
        return

    print("[MAKE] SNP dataset")
    n = len(meta)
    freqs = RNG.uniform(0.05, 0.5, size=N_SNPS)
    X = np.zeros((n, N_SNPS), dtype=np.int8)
    for j, p in enumerate(freqs):
        X[:, j] = RNG.binomial(2, p, size=n)

    y = (meta["label"].str.lower() == "schizophrenia").astype(int).values

    np.save(X_path, X)
    np.save(y_path, y)
    pd.DataFrame({"sample_id": meta["sample_id"]}).to_csv(list_path, index=False)

    print(f"[OK] SNP dataset saved (X={X.shape}, y={y.shape})")


# ---------- MAIN ----------
def main():
    print("=== Synthetic dataset generator ===")

    meta = save_metadata()
    save_genes(meta)
    save_brain(meta)
    save_snps(meta)

    print("\nAll done! Files in:", OUT_DIR)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate several patient PCA-upload files (gene, brain, snp) that give
different model outputs for testing the Streamlit upload/predict flow.

Writes outputs under 'uploads/' like:
  uploads/patient_01_gene_pca.npy
  uploads/patient_01_brain_pca.npy
  uploads/patient_01_snp_pca.npy
  uploads/patient_01_fused.npy

Also prints predictions and writes results/patient_variants_summary.csv

Usage:
  python scripts/generate_patient_variants.py --model models/fusion_rf.pkl --out-dir uploads --n 5
"""
import argparse
from pathlib import Path
import numpy as np
import joblib
import pandas as pd
import os
import sys

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def split_and_save(fused, outdir: Path, idx:int, counts):
    """Split fused into gene/brain/snp according to counts and save .npy files."""
    g = counts['gene']; b = counts['brain']; s = counts['snp']
    assert fused.size == g + b + s
    gvec = fused[:g].reshape(1, -1)
    bvec = fused[g:g+b].reshape(1, -1)
    svec = fused[g+b:].reshape(1, -1)
    # filenames
    fn_base = f"patient_{idx:02d}"
    np.save(outdir / (fn_base + "_gene_pca.npy"), gvec)
    np.save(outdir / (fn_base + "_brain_pca.npy"), bvec)
    np.save(outdir / (fn_base + "_snp_pca.npy"), svec)
    np.save(outdir / (fn_base + "_fused.npy"), fused.reshape(1, -1))
    return str(outdir / (fn_base + "_gene_pca.npy")), str(outdir / (fn_base + "_brain_pca.npy")), str(outdir / (fn_base + "_snp_pca.npy"))

def generate_variants(model, total_dim, n_variants=5, seed=0):
    """
    Generate fused vectors:
     - If model.feature_importances_ exists, use it as a direction to move input.
     - Create variants with negative, small-positive, medium-positive, large-positive, noisy.
    """
    rng = np.random.default_rng(seed)
    base = np.zeros(total_dim, dtype=float)

    imp = None
    if hasattr(model, "feature_importances_"):
        imp = np.array(getattr(model, "feature_importances_")).astype(float)
        if imp.size != total_dim:
            # if mismatch, use top portion or pad zeros
            if imp.size > total_dim:
                imp = imp[:total_dim]
            else:
                pad = np.zeros(total_dim - imp.size, dtype=float)
                imp = np.concatenate([imp, pad])
        # normalize
        if imp.max() > 0:
            imp = imp / (imp.max())
        else:
            imp = None

    variants = []
    # scales to try (negative to positive)
    scales = np.linspace(-2.0, 2.0, min(n_variants,5))
    # If user asked for more than 5, add random variants
    while len(scales) < n_variants:
        scales = np.concatenate([scales, [1.0 + (len(scales)-4)*0.5]])

    for i in range(n_variants):
        s = float(scales[i])
        if imp is not None:
            # Create a vector aligned with importances, add a small structured noise
            direction = imp.copy()
            # emphasize top features slightly
            top_mask = np.zeros_like(direction)
            top_k = max(1, int(0.02 * total_dim))  # top 2%
            top_idx = np.argsort(direction)[-top_k:]
            top_mask[top_idx] = 1.0
            vec = s * direction * (1.0 + 0.3 * top_mask)
            # add low-variance noise scaled by s sign
            noise = rng.normal(scale=0.05 * (1 + abs(s)), size=total_dim)
            vec = vec + noise
        else:
            # fallback: random vector with controlled norm
            vec = rng.normal(scale=0.2*(1+abs(s)), size=total_dim) + s*0.1
        # small clipping to avoid extreme values
        vec = np.clip(vec, -10.0, 10.0)
        variants.append(vec.astype(float))

    return variants

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to trained fusion model (.pkl)")
    p.add_argument("--out-dir", default="uploads", help="Where to save generated .npy files")
    p.add_argument("--n", type=int, default=5, help="Number of variants to create")
    p.add_argument("--counts", type=str, default='{"gene":200,"brain":500,"snp":50}', help="JSON string for PCA counts")
    p.add_argument("--seed", type=int, default=0, help="RNG seed")
    args = p.parse_args()

    outdir = Path(args.out_dir)
    ensure_dir(outdir)
    try:
        model = joblib.load(args.model)
    except Exception as e:
        print("ERROR: could not load model:", e, file=sys.stderr)
        sys.exit(1)

    import json
    counts = json.loads(args.counts)
    total = counts['gene'] + counts['brain'] + counts['snp']

    # generate candidate fused vectors
    variants = generate_variants(model, total, n_variants=args.n, seed=args.seed)

    rows = []
    for i, fused in enumerate(variants, start=1):
        gene_path, brain_path, snp_path = split_and_save(fused, outdir, i, counts)
        # predict using model (if possible)
        try:
            X = fused.reshape(1, -1)
            if hasattr(model, "predict_proba"):
                prob = float(model.predict_proba(X)[0,1])
            else:
                # fallback: use decision_function if available then logistic-like scaling; else 0/1
                if hasattr(model, "decision_function"):
                    score = float(model.decision_function(X)[0])
                    # map via sigmoid
                    prob = 1.0 / (1.0 + np.exp(-score))
                else:
                    pred = int(model.predict(X)[0])
                    prob = float(pred)
        except Exception as e:
            prob = None
        rows.append({
            "patient_id": f"patient_{i:02d}",
            "gene_pca": gene_path,
            "brain_pca": brain_path,
            "snp_pca": snp_path,
            "pred_prob": prob
        })
        print(f"Generated patient_{i:02d} → prob={prob}")

    # write summary CSV
    ensure_dir(Path("results"))
    df = pd.DataFrame(rows)
    df.to_csv(Path("results") / "patient_variants_summary.csv", index=False)
    print("Wrote results/patient_variants_summary.csv")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
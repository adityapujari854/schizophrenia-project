#!/usr/bin/env python3
"""
check_recovery.py

Assess recovery of injected signal features.

Modes:
  - PCA mapping mode (default): inspects results/fusion_top_permutation_importance.csv
    and results/fusion_feature_component_mapping.csv to see whether injected originals
    appear among top-N originals of top-K components.

  - RAW mode (--raw): loads a trained model (default models/fusion_rf.pkl) and the
    concatenated raw X (X_genes.npy, X_brain.npy, X_snps.npy) from data-dir and computes
    feature importances (model.feature_importances_ or permutation_importance fallback).
    Then checks whether the top-ranked raw features correspond to injected indices.

Usage (PowerShell):
  # PCA-mode (default)
  python .\scripts\check_recovery.py --data-dir .\data\synthetic_signal_stronger --top-k 50 --top-n 10

  # RAW-mode (direct feature importances; supply model to inspect)
  python .\scripts\check_recovery.py --data-dir .\data\synthetic_signal_stronger --raw --model models\fusion_rf.pkl --top-features 500

Outputs:
  - results/recovery_matches.csv
  - results/recovery_report.txt
  - (raw mode) results/raw_feature_importances.csv
"""
from pathlib import Path
import argparse
import pandas as pd
import json
import numpy as np
import sys
from collections import defaultdict

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="data/synthetic_signal_stronger", help="Signal dataset dir (has signal_summary.json and X/Y files)")
    p.add_argument("--results-dir", type=str, default="results", help="Results output dir")
    p.add_argument("--top-k", type=int, default=50, help="Top-K components to inspect (PCA mode)")
    p.add_argument("--top-n", type=int, default=10, help="Top-N originals per component (PCA mode)")
    p.add_argument("--raw", action="store_true", help="Raw mode: use model/feature importances on concatenated raw features (no PCA)")
    p.add_argument("--model", type=str, default="models/fusion_rf.pkl", help="Model file to use in raw mode (joblib)")
    p.add_argument("--top-features", type=int, default=1000, help="How many top raw features to inspect in raw mode")
    p.add_argument("--perm-n-jobs", type=int, default=-1, help="n_jobs for permutation importance fallback")
    p.add_argument("--perm-n-repeats", type=int, default=10, help="n_repeats for permutation importance fallback")
    return p.parse_args()

def load_signal_summary(data_dir: Path):
    p = data_dir / "signal_summary.json"
    if not p.exists():
        raise FileNotFoundError(f"signal_summary.json not found in {data_dir}")
    with open(p, "r") as fh:
        return json.load(fh)

def pca_mode(args, summary, results_dir: Path):
    perm_path = results_dir / "fusion_top_permutation_importance.csv"
    map_path = results_dir / "fusion_feature_component_mapping.csv"
    missing = [str(p) for p in (perm_path, map_path) if not p.exists()]
    if missing:
        raise FileNotFoundError("PCA-mode required files missing: " + ", ".join(missing))

    perm = pd.read_csv(perm_path)
    mapping = pd.read_csv(map_path)

    top_components = perm["feature"].astype(str).tolist()[: args.top_k]

    gene_injected = set(summary.get("gene_signal_examples", []) or summary.get("signal_genes", []))
    brain_injected = set(summary.get("brain_signal_examples", []) or summary.get("signal_brain", []))
    snp_injected = set(summary.get("snp_signal_examples", []) or summary.get("signal_snps", []))

    matches = []
    modality_hits = defaultdict(set)
    comp_counts = defaultdict(int)

    for comp in top_components:
        rows = mapping[mapping["feature_component"] == comp].sort_values("rank").head(args.top_n)
        for _, r in rows.iterrows():
            orig_mod = str(r["orig_mod"]).lower()
            try:
                orig_idx = int(r["orig_index"])
            except Exception:
                try:
                    orig_idx = int(float(r["orig_index"]))
                except Exception:
                    continue
            is_injected = False
            if orig_mod.startswith("gene"):
                if orig_idx in gene_injected:
                    is_injected = True
                    modality_hits["genes"].add(orig_idx)
            elif orig_mod.startswith("brain"):
                if orig_idx in brain_injected:
                    is_injected = True
                    modality_hits["brain"].add(orig_idx)
            elif orig_mod.startswith("snp"):
                if orig_idx in snp_injected:
                    is_injected = True
                    modality_hits["snps"].add(orig_idx)

            matches.append({
                "component": comp,
                "orig_mod": orig_mod,
                "orig_index": orig_idx,
                "loading_abs": float(r.get("loading_abs", 0)),
                "rank_within_component": int(r.get("rank", 0)),
                "is_injected": bool(is_injected)
            })
            if is_injected:
                comp_counts[comp] += 1

    matches_df = pd.DataFrame(matches)
    matches_df.to_csv(results_dir / "recovery_matches.csv", index=False)

    total_injected_genes = len(gene_injected)
    total_injected_brain = len(brain_injected)
    total_injected_snps = len(snp_injected)
    recovered_genes = len(modality_hits.get("genes", []))
    recovered_brain = len(modality_hits.get("brain", []))
    recovered_snps = len(modality_hits.get("snps", []))
    comps_with_hits = sum(1 for c in top_components if comp_counts.get(c, 0) > 0)

    report_lines = []
    report_lines.append("=== Recovery report (PCA-mode) ===")
    report_lines.append(f"Data dir: {args.data_dir}")
    report_lines.append(f"Results dir: {args.results_dir}")
    report_lines.append(f"Top-K components inspected: {len(top_components)} (K={args.top_k})")
    report_lines.append(f"Top-N originals per component inspected: {args.top_n}")
    report_lines.append("")
    report_lines.append("Injected counts (from signal_summary.json):")
    report_lines.append(f"  genes injected: {total_injected_genes}")
    report_lines.append(f"  brain indices injected: {total_injected_brain}")
    report_lines.append(f"  snps injected: {total_injected_snps}")
    report_lines.append("")
    report_lines.append("Recovered (unique injected indices found among inspected top-N originals):")
    report_lines.append(f"  genes recovered: {recovered_genes} ({(recovered_genes/total_injected_genes*100) if total_injected_genes else 0:.1f}%)")
    report_lines.append(f"  brain recovered: {recovered_brain} ({(recovered_brain/total_injected_brain*100) if total_injected_brain else 0:.1f}%)")
    report_lines.append(f"  snps recovered: {recovered_snps} ({(recovered_snps/total_injected_snps*100) if total_injected_snps else 0:.1f}%)")
    report_lines.append("")
    report_lines.append(f"Number of top-{args.top_k} components that contained >=1 injected original among top-{args.top_n}: {comps_with_hits}")
    report_lines.append("")
    report_lines.append("Examples of recovered gene indices (up to 50):")
    report_lines.append(", ".join(map(str, sorted(list(modality_hits.get("genes", [])))[:50])) or "(none)")
    report_lines.append("")
    report_lines.append("Examples of recovered brain indices (up to 50):")
    report_lines.append(", ".join(map(str, sorted(list(modality_hits.get("brain", [])))[:50])) or "(none)")
    report_lines.append("")
    report_lines.append("Examples of recovered snp indices (up to 50):")
    report_lines.append(", ".join(map(str, sorted(list(modality_hits.get("snps", [])))[:50])) or "(none)")

    report_text = "\n".join(report_lines)
    (results_dir / "recovery_report.txt").write_text(report_text)
    print(report_text)


def raw_mode(args, summary, results_dir: Path):
    # load model
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    import joblib
    model = joblib.load(model_path)

    data_dir = Path(args.data_dir)
    # load raw modality arrays (must exist)
    Xg_p = data_dir / "X_genes.npy"
    Xb_p = data_dir / "X_brain.npy"
    Xs_p = data_dir / "X_snps.npy"
    for p in (Xg_p, Xb_p, Xs_p):
        if not p.exists():
            raise FileNotFoundError(f"Missing raw feature file: {p}")

    Xg = np.load(str(Xg_p))
    Xb = np.load(str(Xb_p))
    Xs = np.load(str(Xs_p))
    y = np.load(str(data_dir / "y_genes.npy"))

    n_g = Xg.shape[1]
    n_b = Xb.shape[1]
    n_s = Xs.shape[1]

    # Build fusion X ordering: genes | brain | snps (same as training)
    Xf = np.hstack([Xg, Xb, Xs])
    print(f"[INFO] Raw fusion X shape: {Xf.shape} (g={n_g}, b={n_b}, s={n_s})")

    # Try model.feature_importances_ first (fast)
    feat_imp = None
    if hasattr(model, "feature_importances_"):
        try:
            feat_imp = np.asarray(model.feature_importances_, dtype=float)
            if feat_imp.shape[0] != Xf.shape[1]:
                print("[WARN] model.feature_importances_ length does not match X features; ignoring it.")
                feat_imp = None
        except Exception:
            feat_imp = None

    # fallback: permutation importance on a subset (to save time)
    if feat_imp is None:
        print("[INFO] Using permutation_importance fallback (may be slow).")
        from sklearn.inspection import permutation_importance
        # pick subset for permutation (up to 2000 samples)
        n_sub = min(2000, Xf.shape[0])
        rng = np.random.RandomState(42)
        idx = rng.choice(Xf.shape[0], size=n_sub, replace=False)
        X_sub = Xf[idx]
        y_sub = y[idx]
        res = permutation_importance(model, X_sub, y_sub, n_repeats=args.perm_n_repeats, n_jobs=args.perm_n_jobs, scoring="roc_auc")
        feat_imp = res.importances_mean
        # store std as well in results
        feat_imp_std = res.importances_std
    else:
        feat_imp_std = np.zeros_like(feat_imp)

    # build importance table
    inds = np.argsort(feat_imp)[::-1]
    top_k = min(args.top_features, Xf.shape[1])
    top_inds = inds[:top_k]
    rows = []
    for rank, fi in enumerate(top_inds, start=1):
        if fi < n_g:
            mod = "genes"
            orig_idx = int(fi)
        elif fi < n_g + n_b:
            mod = "brain"
            orig_idx = int(fi - n_g)
        else:
            mod = "snps"
            orig_idx = int(fi - n_g - n_b)
        rows.append({
            "feature_index": int(fi),
            "modality": mod,
            "orig_index": int(orig_idx),
            "importance_mean": float(feat_imp[fi]),
            "importance_std": float(feat_imp_std[fi]) if feat_imp_std is not None else 0.0,
            "rank": rank
        })
    imp_df = pd.DataFrame(rows)
    imp_df.to_csv(results_dir / "raw_feature_importances.csv", index=False)

    # load injected indices
    gene_injected = set(summary.get("gene_signal_examples", []) or summary.get("signal_genes", []))
    brain_injected = set(summary.get("brain_signal_examples", []) or summary.get("signal_brain", []))
    snp_injected = set(summary.get("snp_signal_examples", []) or summary.get("signal_snps", []))

    matches = []
    modality_hits = defaultdict(set)
    for _, r in imp_df.iterrows():
        is_injected = False
        if r["modality"] == "genes" and int(r["orig_index"]) in gene_injected:
            is_injected = True
            modality_hits["genes"].add(int(r["orig_index"]))
        if r["modality"] == "brain" and int(r["orig_index"]) in brain_injected:
            is_injected = True
            modality_hits["brain"].add(int(r["orig_index"]))
        if r["modality"] == "snps" and int(r["orig_index"]) in snp_injected:
            is_injected = True
            modality_hits["snps"].add(int(r["orig_index"]))
        matches.append({
            "feature_index": r["feature_index"],
            "modality": r["modality"],
            "orig_index": r["orig_index"],
            "importance_mean": r["importance_mean"],
            "importance_std": r["importance_std"],
            "rank": r["rank"],
            "is_injected": bool(is_injected)
        })
    matches_df = pd.DataFrame(matches)
    matches_df.to_csv(results_dir / "recovery_matches.csv", index=False)

    total_injected_genes = len(gene_injected)
    total_injected_brain = len(brain_injected)
    total_injected_snps = len(snp_injected)
    recovered_genes = len(modality_hits.get("genes", []))
    recovered_brain = len(modality_hits.get("brain", []))
    recovered_snps = len(modality_hits.get("snps", []))

    report_lines = []
    report_lines.append("=== Recovery report (RAW-mode) ===")
    report_lines.append(f"Data dir: {args.data_dir}")
    report_lines.append(f"Model: {args.model}")
    report_lines.append(f"Results dir: {args.results_dir}")
    report_lines.append(f"Top raw features inspected: {top_k}")
    report_lines.append("")
    report_lines.append("Injected counts (from signal_summary.json):")
    report_lines.append(f"  genes injected: {total_injected_genes}")
    report_lines.append(f"  brain indices injected: {total_injected_brain}")
    report_lines.append(f"  snps injected: {total_injected_snps}")
    report_lines.append("")
    report_lines.append("Recovered (unique injected indices found among top raw features):")
    report_lines.append(f"  genes recovered: {recovered_genes} ({(recovered_genes/total_injected_genes*100) if total_injected_genes else 0:.1f}%)")
    report_lines.append(f"  brain recovered: {recovered_brain} ({(recovered_brain/total_injected_brain*100) if total_injected_brain else 0:.1f}%)")
    report_lines.append(f"  snps recovered: {recovered_snps} ({(recovered_snps/total_injected_snps*100) if total_injected_snps else 0:.1f}%)")
    report_lines.append("")
    report_lines.append("Examples of recovered gene indices (up to 50):")
    report_lines.append(", ".join(map(str, sorted(list(modality_hits.get("genes", [])))[:50])) or "(none)")
    report_lines.append("")
    report_lines.append("Examples of recovered brain indices (up to 50):")
    report_lines.append(", ".join(map(str, sorted(list(modality_hits.get("brain", [])))[:50])) or "(none)")
    report_lines.append("")

    report_text = "\n".join(report_lines)
    (results_dir / "recovery_report.txt").write_text(report_text)
    print(report_text)
    print(f"[OK] Raw importances saved to: {results_dir/'raw_feature_importances.csv'}")
    print(f"[OK] Matches saved to: {results_dir/'recovery_matches.csv'}")

def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    summary = load_signal_summary(data_dir)

    if args.raw:
        raw_mode(args, summary, results_dir)
    else:
        pca_mode(args, summary, results_dir)

if __name__ == "__main__":
    main()

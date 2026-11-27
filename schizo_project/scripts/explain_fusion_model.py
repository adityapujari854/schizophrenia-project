#!/usr/bin/env python3
"""
explain_fusion_model.py (improved)

Explain a fusion model's feature importance using SHAP (preferred for tree models)
or sklearn's permutation_importance as fallback.

Saves:
 - results/fusion_X_cached.npy (fusion features used for explanation)
 - results/pca_genes.npz, results/pca_brain.npz, results/pca_snps.npz  (components + mean)
 - results/fusion_top_shap_features.csv  (or fusion_top_permutation_importance.csv)
 - results/fusion_feature_component_mapping.csv  (map PCA comp -> top original features)
 - plots/fusion_shap_summary.png, plots/fusion_shap_beeswarm.png (if SHAP)
 - plots/fusion_permutation_importance.png (if fallback)

Usage (PowerShell):
  python .\scripts\explain_fusion_model.py --model models\fusion_rf.pkl --data-dir data\synthetic_signal_stronger --top 100

Optional flags:
  --n-gene-pca / --n-brain-pca / --n-snp-pca  (defaults match training script: 200/500/50)
  --quick : subsample to 2000 rows for speed
  --force-interventional : use TreeExplainer(feature_perturbation='interventional', check_additivity=False) fallback
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import joblib
import json
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance

RNG = 42

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--data-dir", type=str, default="data/synthetic")
    p.add_argument("--top", type=int, default=100)
    p.add_argument("--n-gene-pca", type=int, default=200)
    p.add_argument("--n-brain-pca", type=int, default=500)
    p.add_argument("--n-snp-pca", type=int, default=50)
    p.add_argument("--quick", action="store_true", help="Use subsample (2000) to speed up")
    p.add_argument("--force-interventional", action="store_true", help="Use TreeExplainer interventional fallback early")
    return p.parse_args()

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_base_arrays(data_dir: Path):
    Xg = np.load(data_dir / "X_genes.npy")
    Xb = np.load(data_dir / "X_brain.npy")
    Xs = np.load(data_dir / "X_snps.npy")
    y  = np.load(data_dir / "y_genes.npy")
    return Xg, Xb, Xs, y

def fit_scale_pca(X, n_comp, cache_path: Path):
    """Fit StandardScaler + PCA on X and save components+mean; return transformed X and (pca, scaler)"""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    n_comp = min(n_comp, Xs.shape[1])
    pca = PCA(n_components=n_comp, random_state=RNG)
    Xp = pca.fit_transform(Xs)
    # Save components and mean and scaler mean/scale into npz
    np.savez_compressed(cache_path, components=pca.components_, mean=pca.mean_, scaler_mean=scaler.mean_, scaler_scale=scaler.scale_)
    return Xp, (scaler, pca)

def load_or_fit_scale_pca(X, n_comp, cache_path: Path):
    if cache_path.exists():
        npz = np.load(cache_path)
        comps = npz["components"]
        mean = npz["mean"]
        scaler_mean = npz["scaler_mean"]
        scaler_scale = npz["scaler_scale"]
        # Recompute transform: (X - scaler_mean)/scaler_scale then dot with comps.T
        Xs = (X - scaler_mean) / scaler_scale
        Xp = np.dot(Xs - mean, comps.T) if mean.size else np.dot(Xs, comps.T)
        # If shapes mismatch (e.g., different n_comp), fallback to fitting anew
        if Xp.shape[1] != min(n_comp, X.shape[1]):
            return fit_scale_pca(X, n_comp, cache_path)
        return Xp, (comps, mean)
    else:
        return fit_scale_pca(X, n_comp, cache_path)

def build_fusion_X(data_dir: Path, results_dir: Path, n_gene_pca, n_brain_pca, n_snp_pca, quick=False):
    # load raw arrays
    Xg, Xb, Xs, y = load_base_arrays(data_dir)
    if quick:
        n_sub = min(2000, Xg.shape[0])
        idx = np.random.RandomState(RNG).choice(Xg.shape[0], size=n_sub, replace=False)
        Xg, Xb, Xs, y = Xg[idx], Xb[idx], Xs[idx], y[idx]
    # cache file paths
    pca_g_path = results_dir / "pca_genes.npz"
    pca_b_path = results_dir / "pca_brain.npz"
    pca_s_path = results_dir / "pca_snps.npz"
    safe_mkdir(results_dir)
    # fit/load transforms
    Xg_p, _ = load_or_fit_scale_pca(Xg, n_gene_pca, pca_g_path)
    Xb_p, _ = load_or_fit_scale_pca(Xb, n_brain_pca, pca_b_path)
    Xs_p, _ = load_or_fit_scale_pca(Xs, n_snp_pca, pca_s_path)
    Xf = np.hstack([Xg_p, Xb_p, Xs_p])
    # persist fusion X for reuse
    np.save(results_dir / "fusion_X_cached.npy", Xf)
    # save a small metadata file describing splits
    meta = {"n_samples": int(Xf.shape[0]), "gene_pca": Xg_p.shape[1], "brain_pca": Xb_p.shape[1], "snp_pca": Xs_p.shape[1]}
    with open(results_dir / "fusion_X_meta.json", "w") as fh:
        json = __import__("json")
        json.dump(meta, fh, indent=2)
    return Xf, y, (Xg_p.shape[1], Xb_p.shape[1], Xs_p.shape[1])

def try_import_shap():
    try:
        import shap
        return shap
    except Exception:
        return None

def plot_shap_summary_and_beeswarm(shap_vals, Xsample, feature_names, plots_dir: Path, top=100):
    import shap
    # summary bar
    plt.figure(figsize=(6,8))
    shap.summary_plot(shap_vals, features=Xsample, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout(); plt.savefig(plots_dir / "fusion_shap_summary.png"); plt.close()
    # beeswarm (use subset for readability)
    plt.figure(figsize=(6,8))
    shap.summary_plot(shap_vals, Xsample, feature_names=feature_names, show=False)
    plt.tight_layout(); plt.savefig(plots_dir / "fusion_shap_beeswarm.png"); plt.close()

def plot_perm_importance(df_sorted, plots_dir: Path):
    plt.figure(figsize=(8,8))
    topn = min(40, df_sorted.shape[0])
    subset = df_sorted.head(topn).iloc[::-1]
    plt.barh(subset["feature"], subset["importance_mean"])
    plt.xlabel("Permutation importance (mean)")
    plt.tight_layout()
    plt.savefig(plots_dir / "fusion_permutation_importance.png")
    plt.close()

def map_components_to_originals(results_dir: Path, top_feature_names, topn_per_comp=10):
    # Load cached PCA npz files to map PCA components back to original features
    gnpz = np.load(results_dir / "pca_genes.npz")
    bnpz = np.load(results_dir / "pca_brain.npz")
    snpnpz = np.load(results_dir / "pca_snps.npz")
    comps_g, mean_g = gnpz["components"], gnpz["mean"]
    comps_b, mean_b = bnpz["components"], bnpz["mean"]
    comps_s, mean_s = snpnpz["components"], snpnpz["mean"]
    # feature name like gPCA_12 or bPCA_34 or sPCA_7
    rows = []
    for fname in top_feature_names:
        pref, idx = fname.split("_")
        idx = int(idx)
        if pref == "gPCA":
            comp = comps_g[idx]
            # top absolute loadings map to original gene indices
            top_idx = np.argsort(np.abs(comp))[-topn_per_comp:][::-1]
            for rank, oi in enumerate(top_idx):
                rows.append({"feature_component": fname, "orig_mod": "genes", "orig_index": int(oi), "loading_abs": float(abs(comp[oi])), "rank": rank+1})
        elif pref == "bPCA":
            comp = comps_b[idx]
            top_idx = np.argsort(np.abs(comp))[-topn_per_comp:][::-1]
            for rank, oi in enumerate(top_idx):
                rows.append({"feature_component": fname, "orig_mod": "brain", "orig_index": int(oi), "loading_abs": float(abs(comp[oi])), "rank": rank+1})
        elif pref == "sPCA":
            comp = comps_s[idx]
            top_idx = np.argsort(np.abs(comp))[-topn_per_comp:][::-1]
            for rank, oi in enumerate(top_idx):
                rows.append({"feature_component": fname, "orig_mod": "snps", "orig_index": int(oi), "loading_abs": float(abs(comp[oi])), "rank": rank+1})
    return pd.DataFrame(rows)

def main():
    args = parse_args()
    model_path = Path(args.model)
    data_dir = Path(args.data_dir)
    results_dir = Path.cwd() / "results"
    plots_dir = Path.cwd() / "plots"
    safe_mkdir(results_dir); safe_mkdir(plots_dir)

    if not model_path.exists():
        print("Model not found:", model_path)
        return

    print("[INFO] Building (or loading) fusion X ...")
    Xf, y, splits = build_fusion_X(data_dir, results_dir, args.n_gene_pca, args.n_brain_pca, args.n_snp_pca, quick=args.quick)
    n_g, n_b, n_s = splits
    print(f"[INFO] Fusion X shape: {Xf.shape} (gPCA={n_g}, bPCA={n_b}, sPCA={n_s})")

    print("[INFO] Loading model:", model_path)
    model = joblib.load(model_path)

    shap = try_import_shap()
    used_shap = False
    if shap is not None:
        print("[INFO] SHAP present. Trying shap.Explainer (auto-detect). This may take a bit.")
        try:
            start = time.time()
            expl = shap.Explainer(model, Xf, feature_perturbation="interventional" if args.force_interventional else "tree_path_dependent")
            # choose subsample for SHAP visuals & speed
            X_for_shap = Xf if Xf.shape[0] <= 2000 else Xf[np.random.RandomState(RNG).choice(Xf.shape[0], 2000, replace=False)]
            sv = expl(X_for_shap)
            # shap.Explainer returns an object with .values; handle binary/class shapes
            shap_vals = sv.values
            # if list-like (multi-output), try select positive-class (index 1)
            if isinstance(shap_vals, list) or (hasattr(shap_vals, "ndim") and shap_vals.ndim == 3 and shap_vals.shape[0] > 1):
                # attempt to select second class
                try:
                    shap_arr = shap_vals[1] if isinstance(shap_vals, list) else shap_vals[1]
                except Exception:
                    shap_arr = shap_vals[0]
            else:
                shap_arr = shap_vals
            # feature names
            feature_names = [f"gPCA_{i}" for i in range(n_g)] + [f"bPCA_{i}" for i in range(n_b)] + [f"sPCA_{i}" for i in range(n_s)]
            # compute mean abs shap
            mean_abs = np.mean(np.abs(shap_arr), axis=0)
            df_shap = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)
            df_shap.head(args.top).to_csv(results_dir / "fusion_top_shap_features.csv", index=False)
            # plots
            try:
                plot_shap_summary_and_beeswarm(shap_arr, X_for_shap, feature_names, plots_dir, top=args.top)
            except Exception as e:
                print("[WARN] SHAP plotting failed:", e)
            print(f"[OK] SHAP saved top {args.top} to results/fusion_top_shap_features.csv (time {time.time()-start:.1f}s)")
            used_shap = True
        except Exception as e:
            print("[WARN] SHAP failed:", e)

    if not used_shap:
        print("[INFO] SHAP not used or failed — running permutation_importance as fallback. This may take time.")
        # use subset for speed
        n_sub = min(2000, Xf.shape[0])
        idx = np.random.RandomState(RNG).choice(Xf.shape[0], size=n_sub, replace=False)
        Xsub, ysub = Xf[idx], y[idx]
        start = time.time()
        res = permutation_importance(model, Xsub, ysub, n_repeats=8, random_state=RNG, n_jobs=-1, scoring="roc_auc")
        imp_mean = res.importances_mean
        imp_std = res.importances_std
        feature_names = [f"gPCA_{i}" for i in range(n_g)] + [f"bPCA_{i}" for i in range(n_b)] + [f"sPCA_{i}" for i in range(n_s)]
        df_imp = pd.DataFrame({"feature": feature_names, "importance_mean": imp_mean, "importance_std": imp_std})
        df_sorted = df_imp.sort_values("importance_mean", ascending=False).reset_index(drop=True)
        df_sorted.head(args.top).to_csv(results_dir / "fusion_top_permutation_importance.csv", index=False)
        plot_perm_importance(df_sorted, plots_dir)
        print(f"[OK] Permutation importance saved (time {time.time()-start:.1f}s)")

    # Map top components to original features for interpretability
    top_features_file = results_dir / ("fusion_top_shap_features.csv" if used_shap else "fusion_top_permutation_importance.csv")
    if top_features_file.exists():
        df_top = pd.read_csv(top_features_file)
        top_feature_names = df_top["feature"].astype(str).tolist()
        mapping_df = map_components_to_originals(results_dir, top_feature_names, topn_per_comp=10)
        mapping_df.to_csv(results_dir / "fusion_feature_component_mapping.csv", index=False)
        print(f"[OK] Saved mapping PCA-comp -> original features to results/fusion_feature_component_mapping.csv")

    print("Done.")

if __name__ == "__main__":
    main()

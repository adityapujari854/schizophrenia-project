#!/usr/bin/env python3
"""
train_fusion_model.py

Train fusion (multimodal) baseline models on gene + brain + SNP features,
with per-modality importance reporting.

Saved outputs (cwd):
  models/
    fusion_logreg.pkl, fusion_rf.pkl, fusion_xgb.pkl (optional)
    scaler_gene.joblib, pca_gene.joblib, ...
  results/
    fusion_metrics.csv
    fusion_feature_importances.csv
    fusion_modality_importance.csv
  plots/
    fusion_<model>_roc.png, fusion_<model>_cm.png

Usage (PowerShell):
  python .\scripts\train_fusion_model.py --data-dir data\synthetic_signal_stronger --quick
  python .\scripts\train_fusion_model.py --data-dir data\synthetic_signal_stronger

Notes:
 - Quick mode uses a 2k subsample and smaller PCA dims for fast iteration.
 - Feature importance aggregation is done on model feature_importances_ (RF/XGB).
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import time
import json
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42

# Optional xgboost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

def parse_args():
    p = argparse.ArgumentParser(description="Train fusion models on gene+brain+snps with modality importances")
    p.add_argument("--data-dir", type=str, default="data/synthetic", help="Directory containing X_genes.npy, X_brain.npy, X_snps.npy, y_genes.npy")
    p.add_argument("--n-gene-pca", type=int, default=200, help="PCA components for genes")
    p.add_argument("--n-brain-pca", type=int, default=500, help="PCA components for brain")
    p.add_argument("--n-snp-pca", type=int, default=50, help="PCA components for snps")
    p.add_argument("--quick", action="store_true", help="Quick mode: subsample and lower PCA dims")
    p.add_argument("--test-size", type=float, default=0.2, help="Test size fraction")
    p.add_argument("--random-state", type=int, default=RANDOM_STATE)
    p.add_argument("--n-estimators", type=int, default=200, help="RandomForest n_estimators")
    p.add_argument("--n-jobs", type=int, default=-1, help="n_jobs for RF/XGB/CV (-1 uses all cores)")
    p.add_argument("--save-pca", action="store_true", help="Save scalers/PCA objects to models/")
    p.add_argument("--no-xgb", action="store_true", help="Skip XGBoost even if installed")
    return p.parse_args()

def ensure_dirs(root: Path):
    for d in ("models", "results", "plots"):
        (root / d).mkdir(parents=True, exist_ok=True)

def load_data(data_dir: Path):
    Xg_path = data_dir / "X_genes.npy"
    Xb_path = data_dir / "X_brain.npy"
    Xs_path = data_dir / "X_snps.npy"
    y_path = data_dir / "y_genes.npy"  # labels assumed shared
    missing = [p for p in (Xg_path, Xb_path, Xs_path, y_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")
    Xg = np.load(Xg_path)
    Xb = np.load(Xb_path)
    Xs = np.load(Xs_path)
    y = np.load(y_path)
    return Xg, Xb, Xs, y

def pca_fit_transform(X, n_comp, name, random_state):
    if n_comp is None or n_comp <= 0:
        return X, None
    print(f"[INFO] PCA {name}: fit n_components={n_comp} on {X.shape}")
    pca = PCA(n_components=min(n_comp, X.shape[1]), random_state=random_state)
    Xp = pca.fit_transform(X)
    ev = float(np.sum(pca.explained_variance_ratio_))
    print(f"[INFO] PCA {name}: output {Xp.shape} (explained var ratio sum: {ev:.4f})")
    return Xp, pca

def run_cv(clf, X, y, folds=5, n_jobs=-1):
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc", n_jobs=n_jobs)
    return float(scores.mean()), float(scores.std())

def save_roc_cm(y_true, y_prob, y_pred, outpath_roc: Path, outpath_cm: Path, title: str):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_true, y_prob):.3f}")
    plt.plot([0,1],[0,1],"--", color="gray")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(title); plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(outpath_roc)
    plt.close()

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    fig, ax = plt.subplots(figsize=(5,4))
    disp.plot(ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(outpath_cm)
    plt.close()

def aggregate_importances(imp_array, split_idxs, modality_names):
    """
    imp_array: 1D array of importances (length = sum of features across modalities after PCA)
    split_idxs: list of ints [n_gene_components, n_brain_components, n_snp_components]
    modality_names: ["genes","brain","snps"]
    returns DataFrame with component-level importances and modality aggregate sums
    """
    rows = []
    start = 0
    for n_comp, mod in zip(split_idxs, modality_names):
        for i in range(n_comp):
            idx = start + i
            rows.append({"feature_index": idx, "modality": mod, "component": i, "importance": float(imp_array[idx])})
        start += n_comp
    df = pd.DataFrame(rows)
    agg = df.groupby("modality")["importance"].sum().reset_index().sort_values("importance", ascending=False)
    return df, agg

def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    root = Path.cwd()
    ensure_dirs(root)
    models_dir = root / "models"
    results_dir = root / "results"
    plots_dir = root / "plots"

    print("[INFO] Loading data from:", data_dir)
    Xg, Xb, Xs, y = load_data(data_dir)
    print(f"[INFO] Shapes loaded: Xg={Xg.shape}, Xb={Xb.shape}, Xs={Xs.shape}, y={y.shape}")
    print("[INFO] Label distribution:", np.bincount(y))

    # quick mode adjustments
    if args.quick:
        print("[INFO] QUICK mode: subsampling 2000 samples and lowering PCA dims")
        n_samples = min(2000, Xg.shape[0])
        idx = np.random.RandomState(args.random_state).choice(Xg.shape[0], size=n_samples, replace=False)
        Xg, Xb, Xs, y = Xg[idx], Xb[idx], Xs[idx], y[idx]
        # lower defaults
        args.n_gene_pca = min(50, args.n_gene_pca)
        args.n_brain_pca = min(100, args.n_brain_pca)
        args.n_snp_pca = min(20, args.n_snp_pca)
        print(f"[INFO] quick shapes: Xg={Xg.shape}, Xb={Xb.shape}, Xs={Xs.shape}, y={y.shape}")

    # Standardize per modality
    scaler_g = StandardScaler()
    scaler_b = StandardScaler()
    scaler_s = StandardScaler()
    Xg_s = scaler_g.fit_transform(Xg)
    Xb_s = scaler_b.fit_transform(Xb)
    Xs_s = scaler_s.fit_transform(Xs)

    # PCA per modality
    Xg_p, pca_g = pca_fit_transform(Xg_s, args.n_gene_pca, "genes", args.random_state)
    Xb_p, pca_b = pca_fit_transform(Xb_s, args.n_brain_pca, "brain", args.random_state)
    Xs_p, pca_s = pca_fit_transform(Xs_s, args.n_snp_pca, "snps", args.random_state)

    # concatenate features
    parts = []
    split_counts = []
    if Xg_p is not None:
        parts.append(Xg_p); split_counts.append(Xg_p.shape[1])
    else:
        parts.append(Xg_s); split_counts.append(Xg_s.shape[1])
    if Xb_p is not None:
        parts.append(Xb_p); split_counts.append(Xb_p.shape[1])
    else:
        parts.append(Xb_s); split_counts.append(Xb_s.shape[1])
    if Xs_p is not None:
        parts.append(Xs_p); split_counts.append(Xs_p.shape[1])
    else:
        parts.append(Xs_s); split_counts.append(Xs_s.shape[1])

    Xf = np.hstack(parts)
    print(f"[INFO] Fusion feature shape: {Xf.shape}  (split counts: {split_counts})")

    # split
    X_train, X_test, y_train, y_test = train_test_split(Xf, y, test_size=args.test_size, stratify=y, random_state=args.random_state)
    print(f"[INFO] Train/test shapes: {X_train.shape}/{X_test.shape}")

    results = []

    # Logistic Regression (CV + final)
    print("[INFO] CV for Logistic Regression ...")
    lr = LogisticRegression(max_iter=1000, solver="saga", random_state=args.random_state)
    mean_auc, std_auc = run_cv(lr, X_train, y_train, folds=5, n_jobs=args.n_jobs)
    print(f"[INFO] LogisticRegression CV AUC = {mean_auc:.4f} ± {std_auc:.4f}")
    results.append({"model": "LogisticRegression_CV", "cv_auc_mean": mean_auc, "cv_auc_std": std_auc})

    lr_pipe = Pipeline([("clf", lr)])
    start = time.time()
    lr_pipe.fit(X_train, y_train)
    tr_time = time.time() - start
    y_prob = lr_pipe.predict_proba(X_test)[:, 1]
    y_pred = lr_pipe.predict(X_test)
    auc_lr = roc_auc_score(y_test, y_prob)
    save_roc_cm(y_test, y_prob, y_pred, plots_dir / "fusion_logreg_roc.png", plots_dir / "fusion_logreg_cm.png", "Fusion LogisticRegression")
    joblib.dump(lr_pipe, models_dir / "fusion_logreg.pkl")
    results.append({"model": "LogisticRegression_test", "test_auc": float(auc_lr), "train_time_s": float(tr_time), "test_n": int(len(y_test))})
    print(f"[OK] logreg test AUC={auc_lr:.4f}, time={tr_time:.1f}s")

    # Random Forest (CV + final)
    print("[INFO] CV for Random Forest ...")
    rf = RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.random_state, n_jobs=args.n_jobs)
    mean_auc_rf, std_auc_rf = run_cv(rf, X_train, y_train, folds=5, n_jobs=args.n_jobs)
    print(f"[INFO] RandomForest CV AUC = {mean_auc_rf:.4f} ± {std_auc_rf:.4f}")
    results.append({"model": "RandomForest_CV", "cv_auc_mean": mean_auc_rf, "cv_auc_std": std_auc_rf})

    start = time.time()
    rf.fit(X_train, y_train)
    tr_time_rf = time.time() - start
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    y_pred_rf = rf.predict(X_test)
    auc_rf = roc_auc_score(y_test, y_prob_rf)
    save_roc_cm(y_test, y_prob_rf, y_pred_rf, plots_dir / "fusion_rf_roc.png", plots_dir / "fusion_rf_cm.png", "Fusion RandomForest")
    joblib.dump(rf, models_dir / "fusion_rf.pkl")
    results.append({"model": "RandomForest_test", "test_auc": float(auc_rf), "train_time_s": float(tr_time_rf), "test_n": int(len(y_test))})
    print(f"[OK] rf test AUC={auc_rf:.4f}, time={tr_time_rf:.1f}s")

    # Feature importances for RF
    if hasattr(rf, "feature_importances_"):
        imp = rf.feature_importances_
        df_imp, df_mod = aggregate_importances(imp, split_counts, ["genes","brain","snps"])
        df_imp.to_csv(results_dir / "fusion_feature_importances.csv", index=False)
        df_mod.to_csv(results_dir / "fusion_modality_importance.csv", index=False)
        print("[INFO] Saved RF feature importances to results/fusion_feature_importances.csv")
        print("[INFO] Saved RF modality-aggregate importances to results/fusion_modality_importance.csv")

    # XGBoost (optional)
    do_xgb = XGBOOST_AVAILABLE and (not args.no_xgb)
    if do_xgb:
        try:
            print("[INFO] XGBoost available: training XGBoost baseline")
            xgb_clf = xgb.XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric="logloss", random_state=args.random_state, n_jobs=args.n_jobs)
            mean_auc_xgb, std_auc_xgb = run_cv(xgb_clf, X_train, y_train, folds=5, n_jobs=args.n_jobs)
            print(f"[INFO] XGBoost CV AUC = {mean_auc_xgb:.4f} ± {std_auc_xgb:.4f}")
            results.append({"model": "XGBoost_CV", "cv_auc_mean": mean_auc_xgb, "cv_auc_std": std_auc_xgb})

            start = time.time()
            xgb_clf.fit(X_train, y_train)
            tr_time_xgb = time.time() - start
            y_prob_xgb = xgb_clf.predict_proba(X_test)[:, 1]
            y_pred_xgb = xgb_clf.predict(X_test)
            auc_xgb = roc_auc_score(y_test, y_prob_xgb)
            save_roc_cm(y_test, y_prob_xgb, y_pred_xgb, plots_dir / "fusion_xgb_roc.png", plots_dir / "fusion_xgb_cm.png", "Fusion XGBoost")
            joblib.dump(xgb_clf, models_dir / "fusion_xgb.pkl")
            results.append({"model": "XGBoost_test", "test_auc": float(auc_xgb), "train_time_s": float(tr_time_xgb), "test_n": int(len(y_test))})
            print(f"[OK] xgb test AUC={auc_xgb:.4f}, time={tr_time_xgb:.1f}s")

            # XGBoost feature importances (if available)
            if hasattr(xgb_clf, "feature_importances_"):
                impx = xgb_clf.feature_importances_
                dfx_imp, dfx_mod = aggregate_importances(impx, split_counts, ["genes","brain","snps"])
                dfx_imp.to_csv(results_dir / "fusion_xgb_feature_importances.csv", index=False)
                dfx_mod.to_csv(results_dir / "fusion_xgb_modality_importance.csv", index=False)
                print("[INFO] Saved XGBoost importances to results/")
        except Exception as e:
            print("[WARN] XGBoost failed:", e)
    else:
        print("[INFO] XGBoost not available or skipped.")

    # Save scalers/pca if requested
    if args.save_pca:
        if scaler_g is not None:
            joblib.dump(scaler_g, models_dir / "scaler_gene.joblib")
        if scaler_b is not None:
            joblib.dump(scaler_b, models_dir / "scaler_brain.joblib")
        if scaler_s is not None:
            joblib.dump(scaler_s, models_dir / "scaler_snps.joblib")
        if pca_g is not None:
            joblib.dump(pca_g, models_dir / "pca_gene.joblib")
        if pca_b is not None:
            joblib.dump(pca_b, models_dir / "pca_brain.joblib")
        if pca_s is not None:
            joblib.dump(pca_s, models_dir / "pca_snps.joblib")
        print("[INFO] Saved scalers and PCA objects to models/")

    # Save results table
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_dir / "fusion_metrics.csv", index=False)
    print("[OK] Fusion training finished. Results saved to", results_dir / "fusion_metrics.csv")
    print("Models saved under:", models_dir)
    print("Plots saved under:", plots_dir)
    print("Done.")

if __name__ == "__main__":
    main()

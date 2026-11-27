#!/usr/bin/env python3
"""
train_gene_model.py (improved / faster / auto-PCA on high-dim)

Train baseline models (Logistic Regression + Random Forest) on gene expression data.

Improvements:
 - automatic PCA when features >> samples (configurable threshold)
 - PCA caching (--cache-pca)
 - configurable n_estimators, n_jobs, max_iter
 - quick mode for fast dev (subsample + lower PCA dims)
 - auto-retry on ConvergenceWarning (increase max_iter and/or enable PCA)
 - clear timing and progress printouts

Usage (PowerShell):
    python .\scripts\train_gene_model.py --data-dir data\synthetic_signal_stronger
    python .\scripts\train_gene_model.py --data-dir data\synthetic_signal_stronger --quick
    python .\scripts\train_gene_model.py --data-dir data\synthetic_signal_stronger --pca-n 200 --cache-pca
"""
from pathlib import Path
import argparse
import time
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, ConfusionMatrixDisplay
from sklearn.exceptions import ConvergenceWarning
import warnings
import json
import sys

# ---- defaults ----
AUTO_PCA_FEATURE_THRESH = 2000  # if feature count > this and no PCA requested, enable PCA automatically
AUTO_PCA_NCOMP = 200           # default n_components when auto-enabling PCA
RANDOM_STATE_DEFAULT = 42

# ----------------- Helpers -----------------
def parse_args():
    p = argparse.ArgumentParser(description="Train gene-only baselines (LR + RF) with optional PCA and auto-retries")
    p.add_argument("--data-dir", type=str, default="data/synthetic", help="Directory containing X_genes.npy and y_genes.npy")
    p.add_argument("--quick", action="store_true", help="Quick dev mode: subsample and lower PCA dims")
    p.add_argument("--test-size", type=float, default=0.2, help="Test fraction")
    p.add_argument("--random-state", type=int, default=RANDOM_STATE_DEFAULT)
    p.add_argument("--pca-n", type=int, default=0, help="If >0, apply PCA to reduce features to this many components")
    p.add_argument("--force-no-auto-pca", action="store_true", help="Do not enable automatic PCA even if features are many")
    p.add_argument("--cache-pca", action="store_true", help="Save/load PCA transform to results/pca_gene.npz to reuse")
    p.add_argument("--n-estimators", type=int, default=100, help="RandomForest n_estimators")
    p.add_argument("--n-jobs", type=int, default=-1, help="n_jobs for RandomForest (use -1 for all cores)")
    p.add_argument("--max-iter", type=int, default=500, help="Max iterations for LogisticRegression (will auto-increase on retry)")
    p.add_argument("--verbose", action="store_true", help="Verbose prints")
    return p.parse_args()

def ensure_dirs(root: Path):
    for d in ("models", "results", "plots"):
        (root / d).mkdir(parents=True, exist_ok=True)

def vprint(s: str, verbose: bool):
    if verbose:
        print(s)
    else:
        # still print info-level messages
        print(s)

def load_data(data_dir: Path):
    Xp = data_dir / "X_genes.npy"
    yp = data_dir / "y_genes.npy"
    if not Xp.exists() or not yp.exists():
        raise FileNotFoundError(f"Missing gene data files in {data_dir}. Expect X_genes.npy and y_genes.npy")
    X = np.load(str(Xp))
    y = np.load(str(yp))
    return X, y

def save_plot_roc(y_true, y_prob, outpath: Path, title: str):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_true, y_prob):.3f}")
    plt.plot([0,1],[0,1],"--", color="gray")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(title); plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(outpath); plt.close()

def save_cm_plot(y_true, y_pred, outpath: Path, title: str):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    fig, ax = plt.subplots(figsize=(5,4))
    disp.plot(ax=ax)
    ax.set_title(title)
    plt.tight_layout(); plt.savefig(outpath); plt.close()

# ----------------- Main -----------------
def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / args.data_dir
    ensure_dirs(project_root)
    models_dir = project_root / "models"
    results_dir = project_root / "results"
    plots_dir = project_root / "plots"

    vprint(f"[INFO] Loading gene features from: {data_dir}", args.verbose)
    X, y = load_data(data_dir)
    vprint(f"[INFO] Shapes: X={X.shape}, y={y.shape}", args.verbose)
    vprint(f"[INFO] Label distribution: {np.bincount(y)}", args.verbose)

    # quick subsample
    if args.quick and X.shape[0] > 2000:
        rng = np.random.RandomState(args.random_state)
        idx = rng.choice(X.shape[0], size=2000, replace=False)
        X, y = X[idx], y[idx]
        vprint(f"[INFO] [QUICK] Subsampled: X={X.shape}, y={y.shape}", True)

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.random_state
    )
    vprint(f"[INFO] Train/test split: {X_train.shape}/{X_test.shape}", True)

    # Decide on auto-PCA when feature count large
    auto_pca_enabled = False
    if args.pca_n and args.pca_n > 0:
        pca_n = min(args.pca_n, X_train.shape[1])
    else:
        # no explicit PCA requested
        if (X_train.shape[1] > AUTO_PCA_FEATURE_THRESH) and (not args.force_no_auto_pca):
            auto_pca_enabled = True
            pca_n = min(AUTO_PCA_NCOMP, X_train.shape[1])
            vprint(f"[INFO] Auto-enabling PCA because feature count={X_train.shape[1]} > {AUTO_PCA_FEATURE_THRESH}. Using n_components={pca_n}", True)
        else:
            pca_n = 0
            vprint("[INFO] PCA disabled (using full feature set)", True)

    # PCA (fit on train only) + optional caching
    pca_obj = None
    pca_path = results_dir / "pca_gene.npz"
    if pca_n and pca_n > 0:
        if args.cache_pca and pca_path.exists():
            vprint("[INFO] Loading cached PCA from results/pca_gene.npz", True)
            npz = np.load(pca_path)
            components = npz["components"]
            mean = npz["mean"]
            # Transform train/test using loaded components
            X_train_centered = X_train - mean
            X_test_centered = X_test - mean
            X_train_p = np.dot(X_train_centered, components.T)
            X_test_p = np.dot(X_test_centered, components.T)
            X_train, X_test = X_train_p, X_test_p
            vprint(f"[INFO] PCA loaded: output shapes {X_train.shape}/{X_test.shape}", True)
        else:
            vprint(f"[INFO] Fitting PCA (n_components={pca_n}) on train data (shape {X_train.shape})", True)
            t0 = time.time()
            pca_obj = PCA(n_components=min(pca_n, X_train.shape[1]), random_state=args.random_state)
            X_train_p = pca_obj.fit_transform(X_train)
            X_test_p = pca_obj.transform(X_test)
            elapsed = time.time() - t0
            vprint(f"[INFO] PCA done in {elapsed:.2f}s; new shapes: {X_train_p.shape}/{X_test_p.shape}", True)
            X_train, X_test = X_train_p, X_test_p
            if args.cache_pca:
                vprint(f"[INFO] Caching PCA components to {pca_path}", True)
                np.savez(pca_path, components=pca_obj.components_, mean=pca_obj.mean_)
    else:
        vprint("[INFO] PCA disabled (using full feature set)", True)

    # Standardize (fit on train)
    vprint("[INFO] Fitting StandardScaler on train data", True)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # containers for results
    results_rows = []

    # ---- Train Logistic Regression with auto-retry on ConvergenceWarning ----
    def train_lr_with_retries(Xtr, ytr, Xte, yte, max_iter, verbose):
        """Try fit LR; if ConvergenceWarning recorded, retry with higher max_iter or enable PCA."""
        # first attempt
        lr = LogisticRegression(max_iter=max_iter, solver="saga", random_state=args.random_state)
        attempt = 1
        current_max_iter = max_iter
        while True:
            vprint(f"[INFO] LR attempt #{attempt}: max_iter={current_max_iter}", verbose)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", ConvergenceWarning)
                t0 = time.time()
                lr.fit(Xtr, ytr)
                train_time = time.time() - t0
                # check for ConvergenceWarning in w
                conv = any(issubclass(x.category, ConvergenceWarning) for x in w)
            if not conv:
                vprint(f"[INFO] LR converged (attempt #{attempt})", verbose)
                return lr, train_time, None
            # else handle retry
            vprint(f"[WARN] LR ConvergenceWarning detected on attempt #{attempt}.", True)
            # strategy: double max_iter once
            if attempt == 1:
                current_max_iter = max(current_max_iter * 2, current_max_iter + 500)
                lr = LogisticRegression(max_iter=current_max_iter, solver="saga", random_state=args.random_state)
                attempt += 1
                vprint(f"[INFO] Retrying LR with max_iter={current_max_iter} ...", True)
                continue
            else:
                # second failure -> signal to caller to use PCA or abort
                return None, None, "convergence"

    vprint("[INFO] Training Logistic Regression ...", True)
    lr_model, lr_time, lr_status = train_lr_with_retries(X_train, y_train, X_test, y_test, args.max_iter, args.verbose)
    # If LR returned status "convergence" (i.e., failed twice) and PCA wasn't yet applied, try enabling PCA automatically
    if lr_model is None and lr_status == "convergence":
        if pca_n == 0 and not args.force_no_auto_pca:
            # enable PCA and retry: fit PCA on original (pre-standardized) train data
            vprint("[INFO] LR failed to converge twice. Enabling PCA (auto) and retrying.", True)
            # need original X_train/X_test before scaling; reload from disk and re-split to ensure consistent preprocessing
            X_full, y_full = load_data(data_dir)
            if args.quick and X_full.shape[0] > 2000:
                rng = np.random.RandomState(args.random_state)
                idx = rng.choice(X_full.shape[0], size=2000, replace=False)
                X_full, y_full = X_full[idx], y_full[idx]
            Xtr_raw, Xte_raw, ytr_raw, yte_raw = train_test_split(X_full, y_full, test_size=args.test_size, stratify=y_full, random_state=args.random_state)
            pca_n2 = min(AUTO_PCA_NCOMP, Xtr_raw.shape[1])
            vprint(f"[INFO] Auto PCA n_components={pca_n2}", True)
            pca2 = PCA(n_components=pca_n2, random_state=args.random_state)
            Xtr_p = pca2.fit_transform(Xtr_raw)
            Xte_p = pca2.transform(Xte_raw)
            # standardize
            scaler2 = StandardScaler()
            Xtr_p = scaler2.fit_transform(Xtr_p)
            Xte_p = scaler2.transform(Xte_p)
            # retry LR with larger max_iter
            lr_model, lr_time, lr_status = train_lr_with_retries(Xtr_p, ytr_raw, Xte_p, yte_raw, max(args.max_iter, 1000), args.verbose)
            if lr_model is None:
                vprint("[ERROR] LR still failed to converge after PCA retry. Aborting LR training.", True)
            else:
                # replace X_train/X_test for downstream RF and plots to ensure consistent eval
                X_train, X_test, y_train, y_test = Xtr_p, Xte_p, ytr_raw, yte_raw
        else:
            vprint("[ERROR] LR failed to converge and auto-PCA disabled. Consider increasing --max-iter or enabling PCA.", True)

    # If LR succeeded, evaluate and save
    if lr_model is not None:
        y_prob_lr = lr_model.predict_proba(X_test)[:, 1]
        y_pred_lr = lr_model.predict(X_test)
        auc_lr = roc_auc_score(y_test, y_prob_lr)
        cm_lr = confusion_matrix(y_test, y_pred_lr)
        joblib.dump(lr_model, models_dir / "gene_logreg.pkl")
        results_rows.append({
            "model": "LogisticRegression",
            "roc_auc": float(auc_lr),
            "train_time_s": float(lr_time),
            "test_n": int(len(y_test))
        })
        vprint(f"[OK] LR done: AUC={auc_lr:.4f}, time={lr_time:.1f}s", True)
        save_plot_roc(y_test, y_prob_lr, plots_dir / "gene_logreg_roc.png", "Gene LogReg ROC")
        save_cm_plot(y_test, y_pred_lr, plots_dir / "gene_logreg_cm.png", "Gene LogReg Confusion Matrix")
    else:
        vprint("[WARN] LR model not available; skipping LR plots/metrics.", True)

    # ----- Random Forest -----
    vprint("[INFO] Training Random Forest ...", True)
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.random_state, n_jobs=args.n_jobs)
    rf.fit(X_train, y_train)
    train_time_rf = time.time() - t0
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    y_pred_rf = rf.predict(X_test)
    auc_rf = roc_auc_score(y_test, y_prob_rf)
    joblib.dump(rf, models_dir / "gene_rf.pkl")
    results_rows.append({
        "model": "RandomForest",
        "roc_auc": float(auc_rf),
        "train_time_s": float(train_time_rf),
        "test_n": int(len(y_test))
    })
    vprint(f"[OK] RF done: AUC={auc_rf:.4f}, time={train_time_rf:.1f}s", True)
    save_plot_roc(y_test, y_prob_rf, plots_dir / "gene_rf_roc.png", "Gene RF ROC")
    save_cm_plot(y_test, y_pred_rf, plots_dir / "gene_rf_cm.png", "Gene RF Confusion Matrix")

    # Save metrics and run record
    dfm = pd.DataFrame(results_rows)
    dfm.to_csv(results_dir / "gene_metrics.csv", index=False)

    run_record = {
        "data_dir": str(data_dir),
        "n_samples": int(X.shape[0]),
        "n_features_after_pca_if_any": int(X_train.shape[1]),
        "pca_n": int(pca_n) if pca_n else 0,
        "n_estimators": int(args.n_estimators),
        "n_jobs": int(args.n_jobs),
        "max_iter": int(args.max_iter),
        "random_state": int(args.random_state),
        "timestamp": time.time()
    }
    with open(results_dir / "gene_run_record.json", "w") as fh:
        json.dump(run_record, fh, indent=2)

    vprint("[OK] Training complete.", True)
    vprint("Saved models -> {}".format(models_dir), True)
    vprint("Saved metrics -> {}".format(results_dir / "gene_metrics.csv"), True)
    vprint("Saved plots -> {}".format(plots_dir), True)


if __name__ == "__main__":
    main()

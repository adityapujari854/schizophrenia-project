#!/usr/bin/env python3
"""
train_brain_model.py

Train baseline models (Logistic Regression + Random Forest) on brain connectivity features.

Usage (PowerShell):
  python .\scripts\train_brain_model.py --data-dir data\synthetic --quick
  python .\scripts\train_brain_model.py --data-dir data\synthetic_signal_stronger
"""

import matplotlib
matplotlib.use("Agg")
from pathlib import Path
import argparse
import time
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, ConfusionMatrixDisplay
import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42

def parse_args():
    p = argparse.ArgumentParser(description="Train brain-only baseline models")
    p.add_argument("--data-dir", type=str, default="data/synthetic", help="Directory with X_brain.npy and y_brain.npy (relative to project root or absolute)")
    p.add_argument("--quick", action="store_true", help="Quick debug mode: subsample and reduce PCA dims")
    p.add_argument("--n-estimators", type=int, default=100, help="RandomForest n_estimators")
    p.add_argument("--n-jobs", type=int, default=-1, help="Number of jobs for RF (use -1 for all cores)")
    p.add_argument("--pca-n", type=int, default=200, help="If >0, apply PCA to reduce features to this many components")
    p.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    p.add_argument("--seed", type=int, default=RANDOM_STATE)
    return p.parse_args()

def ensure_dirs(root: Path):
    for d in ("models", "results", "plots"):
        (root / d).mkdir(parents=True, exist_ok=True)

def quick_status(msg: str):
    print(f"[INFO] {msg}")

def resolve_data_dir(project_root: Path, data_dir_arg: str) -> Path:
    p = Path(data_dir_arg)
    if p.is_absolute():
        return p
    return project_root / p

def load_data(data_dir: Path):
    Xp = data_dir / "X_brain.npy"
    yp = data_dir / "y_brain.npy"
    if not Xp.exists() or not yp.exists():
        raise FileNotFoundError(f"Missing brain data in {data_dir}. Expected X_brain.npy and y_brain.npy")
    X = np.load(str(Xp))
    y = np.load(str(yp))
    return X, y

def fit_and_eval_lr(X_train, y_train, X_test, y_test, seed):
    """Train logistic regression (with scaling) and evaluate."""
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(max_iter=2000, solver="saga", random_state=seed))
    ])
    t0 = time.time()
    pipe.fit(X_train, y_train)
    t = time.time() - t0
    y_prob = pipe.predict_proba(X_test)[:,1]
    y_pred = pipe.predict(X_test)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    return pipe, auc, cm, t

def fit_and_eval_rf(X_train, y_train, X_test, y_test, n_estimators=100, n_jobs=-1, seed=RANDOM_STATE):
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=n_jobs)
    t0 = time.time()
    clf.fit(X_train, y_train)
    t = time.time() - t0
    y_prob = clf.predict_proba(X_test)[:,1]
    y_pred = clf.predict(X_test)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    return clf, auc, cm, t

def plot_save_roc(y_test, y_prob, outpath: Path, title: str):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_test, y_prob):.3f}")
    plt.plot([0,1],[0,1],"--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_save_cm(y_test, y_pred, outpath: Path, title: str):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    fig, ax = plt.subplots(figsize=(5,4))
    disp.plot(ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    ensure_dirs(project_root)
    models_dir = project_root / "models"
    results_dir = project_root / "results"
    plots_dir = project_root / "plots"

    data_dir = resolve_data_dir(project_root, args.data_dir)

    quick_status(f"Loading data from: {data_dir}")
    X, y = load_data(data_dir)
    quick_status(f"Loaded X shape={X.shape}, y shape={y.shape}")
    quick_status(f"Label distribution: {np.bincount(y)}")

    # Quick mode: subsample
    if args.quick:
        n_keep = min(2000, X.shape[0])
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(X.shape[0], size=n_keep, replace=False)
        X = X[idx]
        y = y[idx]
        quick_status(f"[QUICK] Subsampled to {X.shape[0]} samples")

    # Optionally apply PCA to reduce dimensionality
    pca = None
    if args.pca_n and args.pca_n > 0:
        quick_status(f"Applying PCA to reduce to {args.pca_n} components (this may take some time)...")
        pca = PCA(n_components=min(args.pca_n, X.shape[1]), random_state=args.seed)
        t0 = time.time()
        X = pca.fit_transform(X)
        quick_status(f"PCA done in {time.time() - t0:.1f}s; new shape: {X.shape}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, stratify=y, random_state=args.seed)
    quick_status(f"Train/test split: {X_train.shape}/{X_test.shape}")

    # Optional quick CV for RF to estimate performance (small)
    quick_status("Running quick 3-fold CV for RandomForest (AUC) to estimate performance...")
    try:
        rf_cv = RandomForestClassifier(n_estimators=max(10, min(50, args.n_estimators)), random_state=args.seed, n_jobs=args.n_jobs)
        cv_scores = cross_val_score(rf_cv, X_train, y_train, cv=3, scoring="roc_auc", n_jobs=args.n_jobs)
        quick_status(f"RF CV (3-fold) AUC mean={cv_scores.mean():.4f} std={cv_scores.std():.4f}")
    except Exception as e:
        quick_status(f"[WARN] RF CV failed: {e}")

    # --- Logistic Regression final train ---
    quick_status("Training Logistic Regression (may take time)...")
    lr_clf, auc_lr, cm_lr, t_lr = None, None, None, None
    try:
        lr_clf, auc_lr, cm_lr, t_lr = fit_and_eval_lr(X_train, y_train, X_test, y_test, seed=args.seed)
        quick_status(f"LogReg done: test AUC={auc_lr:.4f}, time={t_lr:.1f}s")
        joblib.dump(lr_clf, models_dir / "brain_logreg.pkl")
        # save ROC + CM
        plot_save_roc(y_test, lr_clf.predict_proba(X_test)[:,1], plots_dir / "brain_logreg_roc.png", "Brain LogReg ROC")
        plot_save_cm(y_test, lr_clf.predict(X_test), plots_dir / "brain_logreg_cm.png", "Brain LogReg Confusion Matrix")
    except Exception as e:
        quick_status(f"[ERROR] Logistic Regression failed: {e}")

    # --- Random Forest final train ---
    quick_status(f"Training RandomForest (n_estimators={args.n_estimators}) ...")
    rf_clf, auc_rf, cm_rf, t_rf = None, None, None, None
    try:
        rf_clf, auc_rf, cm_rf, t_rf = fit_and_eval_rf(X_train, y_train, X_test, y_test, n_estimators=args.n_estimators, n_jobs=args.n_jobs, seed=args.seed)
        quick_status(f"RandomForest done: test AUC={auc_rf:.4f}, time={t_rf:.1f}s")
        joblib.dump(rf_clf, models_dir / "brain_rf.pkl")
        plot_save_roc(y_test, rf_clf.predict_proba(X_test)[:,1], plots_dir / "brain_rf_roc.png", "Brain RF ROC")
        plot_save_cm(y_test, rf_clf.predict(X_test), plots_dir / "brain_rf_cm.png", "Brain RF Confusion Matrix")
    except Exception as e:
        quick_status(f"[ERROR] Random Forest failed: {e}")

    # Save metrics
    rows = []
    if auc_lr is not None:
        rows.append({"model":"LogisticRegression","test_auc": float(auc_lr), "train_time_s": float(t_lr), "test_n": int(len(y_test))})
    if auc_rf is not None:
        rows.append({"model":"RandomForest","test_auc": float(auc_rf), "train_time_s": float(t_rf), "test_n": int(len(y_test))})
    if rows:
        pd.DataFrame(rows).to_csv(results_dir / "brain_metrics.csv", index=False)
        quick_status(f"Saved metrics to {results_dir / 'brain_metrics.csv'}")
    else:
        quick_status("[WARN] No metrics to save")

    quick_status("Done.")

if __name__ == "__main__":
    main()

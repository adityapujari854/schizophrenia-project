import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import joblib

ROOT = Path(__file__).resolve().parents[1]
SYN = ROOT / "data" / "synthetic"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
MODELS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

def train_and_save(X, y, prefix):
    valid = (y == 0) | (y == 1)
    X = X[valid]
    y = y[valid]
    if len(np.unique(y)) < 2:
        print(f"[WARN] Not enough classes for {prefix}, skipping")
        return
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    models = {
        "rf": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "logreg": LogisticRegression(max_iter=1000, solver="liblinear")
    }
    results = {}
    best_auc = -1
    best_name = None
    best_model = None
    for name, m in models.items():
        m.fit(X_train, y_train)
        probs = m.predict_proba(X_test)[:, 1] if hasattr(m, "predict_proba") else m.decision_function(X_test)
        auc = float(roc_auc_score(y_test, probs))
        preds = (probs >= 0.5).astype(int)
        rep = classification_report(y_test, preds, output_dict=True)
        results[name] = {"auc": auc, "report": rep}
        if auc > best_auc:
            best_auc = auc
            best_name = name
            best_model = m
    if best_model is not None:
        joblib.dump(best_model, MODELS / f"{prefix}_model_{best_name}.joblib")
    with open(RESULTS / f"{prefix}_synthetic_results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[OK] {prefix} training done. Best: {best_name} (AUC={best_auc:.3f})")

def main():
    # Genes
    Xg_path = SYN / "X_genes.npy"
    yg_path = SYN / "y_genes.npy"
    if Xg_path.exists() and yg_path.exists():
        Xg = np.load(Xg_path)
        yg = np.load(yg_path)
        train_and_save(Xg, yg, "genes")
    else:
        print("[WARN] Synthetic genes files missing, skipping gene training.")
    # Brain
    Xb_path = SYN / "X_brain.npy"
    yb_path = SYN / "y_brain.npy"
    if Xb_path.exists() and yb_path.exists():
        Xb = np.load(Xb_path)
        yb = np.load(yb_path)
        train_and_save(Xb, yb, "brain")
    else:
        print("[WARN] Synthetic brain files missing, skipping brain training.")

if __name__ == "__main__":
    main()
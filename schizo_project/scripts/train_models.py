from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
import joblib

# ...existing code...
# minimal helper training script that uses dataset loader in scripts/dataset.py
from scripts.dataset import load_all  # uses your loader: load_all (schizo_project/scripts/dataset.py)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_PROC = PROJECT_ROOT / "data" / "processed"
OUT_PROC.mkdir(parents=True, exist_ok=True)
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def prepare_gene_matrix(geo_df: pd.DataFrame):
    # GEO is probe-indexed, columns are sample IDs (strings like "GSM...")
    # convert to numeric and transpose -> samples x probes
    geo_numeric = geo_df.apply(pd.to_numeric, errors="coerce")
    # drop probes with too many missing values
    probe_na_frac = geo_numeric.isna().mean(axis=1)
    geo_numeric = geo_numeric.loc[probe_na_frac < 0.2]
    geo_T = geo_numeric.T.fillna(geo_numeric.mean(axis=1))
    return geo_T  # rows = samples, cols = probes

def build_Xy(part_df: pd.DataFrame, gene_X: pd.DataFrame):
    # attempt to align participant IDs with gene sample IDs
    part_ids = set(part_df["participant_id"].astype(str).str.strip())
    geo_ids = set(gene_X.index.astype(str).str.strip())
    inter = part_ids.intersection(geo_ids)
    if len(inter) == 0:
        print("No overlap between participants.tsv IDs and GEO sample IDs.")
        print("Saved processed gene matrix to data/processed/genes/gene_matrix.npy for manual mapping.")
        # save to processed for manual downstream mapping
        (OUT_PROC / "genes").mkdir(parents=True, exist_ok=True)
        np.save(OUT_PROC / "genes" / "gene_matrix.npy", gene_X.values)
        gene_X.to_csv(OUT_PROC / "genes" / "gene_matrix_samples.csv")
        return None, None

    # keep only overlapping samples and get labels from participants
    keep = sorted(list(inter))
    X = gene_X.loc[keep].values
    # create y mapping using participant diagnosis column (normalize)
    part_map = part_df.set_index("participant_id")["diagnosis"].astype(str).str.strip().str.upper().to_dict()
    y = np.array([1 if part_map[s] in ("SCHZ", "SCHIZOPHRENIA") else 0 for s in keep])
    return X, y

def simple_preprocess(X: np.ndarray):
    # variance threshold + scaling
    vt = VarianceThreshold(threshold=1e-2)
    X = vt.fit_transform(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, vt, scaler

def train_and_eval(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    # baseline models
    models = {
        "rf": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "logreg": LogisticRegression(max_iter=1000, solver="liblinear"),
    }
    best_auc = -1
    best_name = None
    best_model = None
    results = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        probs = m.predict_proba(X_test)[:, 1] if hasattr(m, "predict_proba") else m.decision_function(X_test)
        preds = (probs >= 0.5).astype(int)
        auc = roc_auc_score(y_test, probs)
        rep = classification_report(y_test, preds, output_dict=True)
        results[name] = {"auc": float(auc), "report": rep}
        print(f"Model {name}: ROC AUC = {auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            best_name = name
            best_model = m
    # save best
    if best_model is not None:
        joblib.dump(best_model, MODELS_DIR / f"best_model_{best_name}.joblib")
        print(f"Saved best model: {MODELS_DIR / f'best_model_{best_name}.joblib'}")
    # also save results
    (PROJECT_ROOT / "results").mkdir(parents=True, exist_ok=True)
    import json
    with open(PROJECT_ROOT / "results" / "training_results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    return results

def main():
    data = load_all()  # uses your loader: load_all (schizo_project/scripts/dataset.py)
    part_df = data["participants"]
    geo_df = data["geo"]
    if geo_df is None:
        print("No GEO matrix loaded. Run dataset loader or place series matrix at expected path.")
        sys.exit(1)
    gene_X = prepare_gene_matrix(geo_df)
    X, y = build_Xy(part_df, gene_X)
    if X is None:
        sys.exit(0)  # saved processed matrix; user must map samples to participants
    # preprocess
    X_proc, vt, scaler = simple_preprocess(X)
    # save X,y
    np.save(OUT_PROC / "X.npy", X_proc)
    np.save(OUT_PROC / "y.npy", y)
    joblib.dump(vt, MODELS_DIR / "variance_threshold.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    print("Saved processed X.npy and y.npy")
    # train
    results = train_and_eval(X_proc, y)
    print("Training complete. Results saved to results/training_results.json")

if __name__ == "__main__":
    main()
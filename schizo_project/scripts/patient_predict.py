#!/usr/bin/env python3
"""
patient_predict.py (improved, single-sample friendly)

Usage examples:

# Full-dataset mode (use existing/cache PCA or re-fit from full dataset)
python scripts/patient_predict.py --data-dir data/synthetic_fullrun --sample-index 0 --model models/fusion_rf.pkl --out results/patient_pred.csv --verbose

# Single-sample mode (provide single-sample vectors). Requires PCA caches to already exist in results/
python scripts/patient_predict.py --gene-file uploads/patient_gene.npy --brain-file uploads/patient_brain.npy --snp-file uploads/patient_snp.npy --model models/fusion_rf.pkl --out results/patient_pred_single.csv --use-cached-only --verbose

Notes:
 - Single-sample mode WILL NOT FIT PCA (impossible); it either loads cached PCA transforms or uses --pca-* overrides to construct padded/truncated fusion vector.
 - PCA caches are expected in results/pca_gene.npz, results/pca_brain.npz, results/pca_snp.npz and must contain at least "components" and "mean" or "X_pca".
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import joblib
import os
import sys
import time
import traceback
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# ---- logging helpers ----
def log(msg): print(f"[INFO] {msg}")
def warn(msg): print(f"[WARN] {msg}")
def err(msg): print(f"[ERROR] {msg}")

# ---- IO helpers ----
def load_npy_or_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() == ".npy":
        return np.load(str(path))
    else:
        # read CSV -> flatten to 1D if single row/col
        df = pd.read_csv(path, header=None)
        arr = df.values
        # if shape like (n,1) or (1,n) flatten to 1D
        if arr.ndim == 2 and (arr.shape[0] == 1 or arr.shape[1] == 1):
            return arr.ravel()
        return arr

def load_array_from_dir(dirpath: Path, name: str):
    p = dirpath / f"{name}.npy"
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    return np.load(str(p))

# ---- PCA cache helpers ----
def read_pca_cache(cache_path: Path):
    """Return dict-like object if valid; else None."""
    if not cache_path.exists():
        return None
    try:
        with np.load(str(cache_path), allow_pickle=False) as npz:
            keys = set(npz.files)
            # Accept either 'X_pca' (full transformed matrix) OR components+mean
            if "X_pca" in keys or ("components" in keys and "mean" in keys):
                return dict(npz)
            else:
                warn(f"PCA cache {cache_path} missing expected keys; found {keys}")
                return None
    except Exception as e:
        warn(f"Failed to read PCA cache {cache_path}: {e}")
        return None

def safe_remove(path: Path):
    try:
        path.unlink()
        log(f"Removed file: {path}")
        return True
    except PermissionError:
        warn(f"PermissionError removing {path}; file may be locked by another process.")
        return False
    except FileNotFoundError:
        return True
    except Exception as e:
        warn(f"Unexpected error removing {path}: {e}")
        return False

def save_pca_cache(cache_path: Path, components=None, mean=None, X_pca=None):
    """Try to save an npz; if write fails due to lock, write timestamped alt file."""
    try:
        tmp = str(cache_path) + f".tmp{int(time.time()*1000)}"
        if X_pca is not None:
            np.savez(tmp, X_pca=X_pca)
        else:
            np.savez(tmp, components=components, mean=mean)
        # atomic replace if possible
        try:
            os.replace(tmp, str(cache_path))
        except Exception:
            os.rename(tmp, str(cache_path))
        log(f"Wrote PCA cache: {cache_path}")
        return cache_path
    except PermissionError as e:
        warn(f"PermissionError writing cache {cache_path}: {e}. Will attempt alt file.")
        ts = datetime.now().strftime("%Y%m%d%H%M%S%f")
        alt = cache_path.with_name(cache_path.stem + f"_{ts}" + cache_path.suffix)
        try:
            if X_pca is not None:
                np.savez(alt, X_pca=X_pca)
            else:
                np.savez(alt, components=components, mean=mean)
            log(f"Wrote alternate PCA cache: {alt}")
            return alt
        except Exception as e2:
            warn(f"Failed to write alternate cache {alt}: {e2}")
            return None
    except Exception as e:
        warn(f"Failed to write PCA cache {cache_path}: {e}")
        return None

# ---- PCA transform utilities ----
def transform_with_cache_or_components(data, n_components, tag, cache_dir: Path, allow_refit=True):
    """
    If cache exists and contains X_pca => use X_pca (and potentially slice to requested n_components)
    If cache has components+mean => compute transform: (data - mean) @ components.T
    If no valid cache:
      - if allow_refit==True: fit PCA(n_components) on `data` and save X_pca
      - if allow_refit==False: raise error
    Returns transformed array and a dict describing source.
    """
    cache_path = cache_dir / f"pca_{tag}.npz"
    cached = read_pca_cache(cache_path)
    if cached is not None:
        # If 'X_pca' present, we expect full transformed matrix for a dataset; try to adapt to requested n_components.
        if "X_pca" in cached:
            X_pca = cached["X_pca"]
            # ensure correct orientation (n_samples, n_components)
            if X_pca.ndim != 2:
                raise ValueError(f"Cached X_pca shape invalid for {tag}: {X_pca.shape}")
            # If requested n_components smaller than cached, slice columns; if larger, pad with zeros.
            if n_components is None:
                return X_pca, {"source": "cache_X_pca", "cached_components": X_pca.shape[1]}
            else:
                if X_pca.shape[1] >= n_components:
                    return X_pca[:, :n_components], {"source": "cache_X_pca_sliced", "cached_components": X_pca.shape[1]}
                else:
                    # pad
                    pad = np.zeros((X_pca.shape[0], n_components - X_pca.shape[1]))
                    return np.hstack((X_pca, pad)), {"source": "cache_X_pca_padded", "cached_components": X_pca.shape[1]}
        else:
            # components + mean present -> perform linear transform
            comps = cached.get("components")
            mean = cached.get("mean")
            if comps is None or mean is None:
                raise ValueError(f"Cache for {tag} invalid (missing components/mean).")
            # comps expected shape (n_components_cached, n_features) - sklearn stores components_ as (n_components, n_features)
            # we transform: (data - mean) @ comps.T  -> resulting shape (n_samples, n_components_cached)
            comps = np.array(comps)
            mean = np.array(mean)
            X_centered = data - mean
            X_pca = X_centered @ comps.T
            if n_components is None:
                return X_pca, {"source": "cache_components", "cached_components": X_pca.shape[1]}
            else:
                if X_pca.shape[1] >= n_components:
                    return X_pca[:, :n_components], {"source": "cache_components_sliced", "cached_components": X_pca.shape[1]}
                else:
                    pad = np.zeros((X_pca.shape[0], n_components - X_pca.shape[1]))
                    return np.hstack((X_pca, pad)), {"source": "cache_components_padded", "cached_components": X_pca.shape[1]}
    # no valid cache
    if not allow_refit:
        raise RuntimeError(f"No valid PCA cache found for '{tag}' and refit not allowed (single-sample / use-cached-only).")
    # Fit PCA on provided `data` (only used when full-dataset mode)
    n_comp = n_components if (n_components is not None and n_components > 0) else min(data.shape[0], data.shape[1])
    if n_comp <= 0:
        raise ValueError("n_components must be > 0 to fit PCA.")
    log_msg = f"Fitting PCA (n_components={n_comp}) for tag '{tag}' on data shape {data.shape}"
    log(log_msg)
    pca = PCA(n_components=n_comp, random_state=42)
    X_pca = pca.fit_transform(data)
    # Save either components+mean and X_pca to cache for full-dataset convenience
    save_pca_cache(cache_dir / f"pca_{tag}.npz", components=pca.components_, mean=pca.mean_, X_pca=X_pca)
    return X_pca, {"source": "fitted", "fitted_components": X_pca.shape[1]}

# ---- fusion builder ----
def build_fusion(Xg_pca, Xb_pca, Xs_pca, gene_n, brain_n, snp_n):
    parts = []
    for arr, expected in [(Xg_pca, gene_n), (Xb_pca, brain_n), (Xs_pca, snp_n)]:
        if expected is None:
            parts.append(arr)
            continue
        # ensure arr 2D
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        # pad or truncate columns
        if arr.shape[1] < expected:
            pad = np.zeros((arr.shape[0], expected - arr.shape[1]))
            parts.append(np.hstack([arr, pad]))
        else:
            parts.append(arr[:, :expected])
    return np.hstack(parts)

# ---- infer PCA counts from mapping CSV ----
def infer_pca_from_mapping(mapping_csv: Path):
    if not mapping_csv.exists():
        return None, None, None
    df = pd.read_csv(mapping_csv, dtype=str)
    # choose column name containing feature string
    col = None
    for c in ["feature_component", "feature", "component"]:
        if c in df.columns:
            col = c
            break
    if col is None:
        col = df.columns[0]
    s = df[col].astype(str)
    g = int(s.str.startswith("gPCA").sum())
    b = int(s.str.startswith("bPCA").sum())
    s_ = int(s.str.startswith("sPCA").sum())
    return g, b, s_

# ---- main ----
def main():
    p = argparse.ArgumentParser(description="Predict single patient using fusion model (robust PCA handling)")
    # inputs: either data-dir (full) or per-sample files
    p.add_argument("--data-dir", type=str, help="Directory with X_genes.npy, X_brain.npy, X_snps.npy (full-dataset mode)")
    p.add_argument("--gene-file", type=str, help="Single-sample gene vector (.npy or .csv)")
    p.add_argument("--brain-file", type=str, help="Single-sample brain vector (.npy or .csv)")
    p.add_argument("--snp-file", type=str, help="Single-sample snp vector (.npy or .csv)")
    p.add_argument("--model", type=str, required=True, help="Trained fusion model (joblib .pkl)")
    p.add_argument("--sample-index", type=int, default=0, help="Index for dataset mode (ignored for single-sample files)")
    p.add_argument("--out", type=str, required=True, help="Output CSV for prediction")
    p.add_argument("--use-cached-only", action="store_true", help="Do NOT fit PCA if cache missing (required for single-sample predictions)")
    p.add_argument("--pca-gene", type=int, help="Override gene PCA component count")
    p.add_argument("--pca-brain", type=int, help="Override brain PCA component count")
    p.add_argument("--pca-snp", type=int, help="Override snp PCA component count")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    model_path = Path(args.model)
    if not model_path.exists():
        err(f"Model file not found: {model_path}")
        sys.exit(1)
    try:
        model = joblib.load(str(model_path))
    except Exception as e:
        err(f"Failed to load model: {e}")
        traceback.print_exc()
        sys.exit(1)
    expected_features = getattr(model, "n_features_in_", None)
    if args.verbose:
        log(f"Loaded model: {model_path}; model expects n_features_in_={expected_features}")

    # Determine mode
    single_sample_mode = any([args.gene_file, args.brain_file, args.snp_file])
    if single_sample_mode:
        # require at least gene+brain (snps optional)
        if not (args.gene_file and args.brain_file):
            err("Single-sample mode requires --gene-file and --brain-file at minimum.")
            sys.exit(1)
        # load single-sample arrays
        try:
            Xg = load_npy_or_csv(Path(args.gene_file))
            Xb = load_npy_or_csv(Path(args.brain_file))
            Xs = None
            if args.snp_file:
                Xs = load_npy_or_csv(Path(args.snp_file))
        except Exception as e:
            err(f"Failed to load single-sample files: {e}")
            sys.exit(1)
        # ensure shapes -> make (1, n_features)
        if Xg.ndim == 1: Xg = Xg.reshape(1, -1)
        if Xb.ndim == 1: Xb = Xb.reshape(1, -1)
        if Xs is not None and Xs.ndim == 1: Xs = Xs.reshape(1, -1)
        data_mode = "single"
        if args.verbose: log(f"Single-sample data shapes: genes={Xg.shape}, brain={Xb.shape}, snps={None if Xs is None else Xs.shape}")
    else:
        # dataset mode: load arrays from data-dir
        if not args.data_dir:
            err("Either --data-dir (dataset mode) or --gene-file/--brain-file (single-sample mode) must be provided.")
            sys.exit(1)
        data_dir = Path(args.data_dir)
        try:
            Xg = load_array_from_dir(data_dir, "X_genes")
            Xb = load_array_from_dir(data_dir, "X_brain")
            Xs = None
            try:
                Xs = load_array_from_dir(data_dir, "X_snps")
            except FileNotFoundError:
                Xs = np.zeros((Xg.shape[0], 0))
        except Exception as e:
            err(f"Failed to load arrays from data-dir: {e}")
            sys.exit(1)
        data_mode = "dataset"
        if args.verbose: log(f"Dataset data shapes: genes={Xg.shape}, brain={Xb.shape}, snps={Xs.shape}")

    # infer PCA counts from mapping CSV if present
    mapping_csv = results_dir / "fusion_feature_component_mapping.csv"
    g_cnt, b_cnt, s_cnt = (None, None, None)
    if mapping_csv.exists():
        try:
            g_cnt, b_cnt, s_cnt = infer_pca_from_mapping(mapping_csv)
            if args.verbose:
                log(f"Inferred PCA counts from mapping CSV: gene={g_cnt}, brain={b_cnt}, snp={s_cnt}")
        except Exception as e:
            warn(f"Failed to infer PCA counts: {e}")

    # apply CLI overrides
    if args.pca_gene is not None: g_cnt = args.pca_gene
    if args.pca_brain is not None: b_cnt = args.pca_brain
    if args.pca_snp is not None: s_cnt = args.pca_snp

    # If model expects a common default (750), set default if missing
    if expected_features in (750, 751, 747) and (g_cnt is None or b_cnt is None or s_cnt is None):
        if args.verbose:
            log(f"[Auto-fix] Model expects {expected_features} features — setting default PCA counts to (200,500,50).")
        g_cnt = g_cnt or 200
        b_cnt = b_cnt or 500
        s_cnt = s_cnt or 50

    # For single-sample mode we MUST NOT attempt to fit PCA on the single sample.
    allow_refit = True
    if data_mode == "single":
        if args.use_cached_only:
            allow_refit = False
        else:
            # force false and warn — fitting PCA on a single sample will fail; require cached PCA
            allow_refit = False
            warn("Single-sample mode: PCA will NOT be refit from single sample. Use existing caches in results/ or provide --pca-* overrides and --use-cached-only.")
    if args.verbose:
        log(f"PCA behavior: allow_refit={allow_refit}")

    # Scaleers/fits:
    if data_mode == "dataset":
        # Fit scalers on full dataset (train-like)
        scaler_g = StandardScaler().fit(Xg)
        scaler_b = StandardScaler().fit(Xb)
        scaler_s = None
        if Xs.size != 0:
            scaler_s = StandardScaler().fit(Xs)
        # transform
        Xg_s = scaler_g.transform(Xg)
        Xb_s = scaler_b.transform(Xb)
        Xs_s = scaler_s.transform(Xs) if (scaler_s is not None) else np.zeros((Xg.shape[0], 0))
    else:
        # single-sample: we cannot fit a scaler reliably; we still can attempt to load scaler params from caches (not implemented).
        # Use StandardScaler() fit on single sample (results in zero-variance) — better approach is to require PCA caches computed on training data
        # We'll standardize by subtracting mean=0 and dividing by 1 (no-op) to avoid zero-division.
        # But this is not ideal: recommended to use full-dataset PCA caches produced during training.
        Xg_s = Xg.astype(float)
        Xb_s = Xb.astype(float)
        Xs_s = Xs.astype(float) if Xs is not None else np.zeros((Xg.shape[0], 0))
        if args.verbose:
            warn("Single-sample mode: no reliable scaler available. Using identity transform (NO standardization).")

    # Now transform with PCA caches or fit if allowed (dataset mode only)
    cache_dir = results_dir
    try:
        Xg_pca, src_g = transform_with_cache_or_components(Xg_s, g_cnt, "gene", cache_dir, allow_refit=allow_refit)
        Xb_pca, src_b = transform_with_cache_or_components(Xb_s, b_cnt, "brain", cache_dir, allow_refit=allow_refit)
        Xs_pca, src_s = transform_with_cache_or_components(Xs_s, s_cnt, "snp", cache_dir, allow_refit=allow_refit)
    except Exception as e:
        err(f"PCA transform failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    if args.verbose:
        log(f"PCA sources: gene={src_g}, brain={src_b}, snp={src_s}")
        log(f"PCA output shapes: gene={Xg_pca.shape}, brain={Xb_pca.shape}, snp={Xs_pca.shape}")

    # Build fusion matrix (full, many samples if dataset mode, or single sample)
    fusion = build_fusion(Xg_pca, Xb_pca, Xs_pca, g_cnt, b_cnt, s_cnt)
    if args.verbose:
        log(f"Built fusion matrix shape {fusion.shape}")

    # Choose sample vector
    if data_mode == "dataset":
        idx = int(args.sample_index)
        if idx < 0 or idx >= fusion.shape[0]:
            err(f"Sample index {idx} out of range (0..{fusion.shape[0]-1})")
            sys.exit(1)
        X_patient = fusion[[idx], :]
    else:
        # single sample: fusion should be shape (1, n_features)
        X_patient = fusion
        if X_patient.shape[0] != 1:
            # If user passed multiple rows in CSV, pick first
            X_patient = X_patient[[0], :]

    # Validate vs model expected features
    if expected_features is not None and X_patient.shape[1] != expected_features:
        warn(f"Fusion input has {X_patient.shape[1]} features, but model expects {expected_features}.")
        # attempt auto-fix: truncate or pad -> allowed, but warn strongly.
        if X_patient.shape[1] > expected_features:
            X_patient = X_patient[:, :expected_features]
            warn(f"[Auto-truncate] Truncated fusion vector to {expected_features} features.")
        else:
            pad = np.zeros((1, expected_features - X_patient.shape[1]))
            X_patient = np.hstack([X_patient, pad])
            warn(f"[Auto-pad] Padded fusion vector to {expected_features} features. This may reduce fidelity.")

    # Prediction
    try:
        proba = model.predict_proba(X_patient)[0, 1]
        label = int(proba > 0.5)
        log(f"Predicted class={label} (prob={proba:.4f})")
    except Exception as e:
        err(f"Model prediction failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Save output
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "mode": data_mode,
        "sample_index": int(args.sample_index) if data_mode == "dataset" else 0,
        "predicted_label": int(label),
        "probability_schizophrenia": float(proba),
        "pca_gene_components": int(g_cnt) if g_cnt is not None else None,
        "pca_brain_components": int(b_cnt) if b_cnt is not None else None,
        "pca_snp_components": int(s_cnt) if s_cnt is not None else None,
        "model_expected_features": int(expected_features) if expected_features is not None else None,
        "timestamp": datetime.now().isoformat(),
        "pca_gene_source": str(src_g),
        "pca_brain_source": str(src_b),
        "pca_snp_source": str(src_s)
    }
    df = pd.DataFrame([rec])
    df.to_csv(outp, index=False)
    log(f"[OK] Wrote prediction to: {outp}")

if __name__ == "__main__":
    main()
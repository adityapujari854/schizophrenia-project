#!/usr/bin/env python3
"""
Quick validator for synthetic dataset files.
Saves nothing; prints informative summary to console.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import sys

ROOT = Path.cwd()
SYN = ROOT / "data" / "synthetic"

files_expected = {
    "X_genes": SYN / "X_genes.npy",
    "y_genes": SYN / "y_genes.npy",
    "gene_samples": SYN / "gene_samples_list.csv",
    "X_brain": SYN / "X_brain.npy",
    "y_brain": SYN / "y_brain.npy",
    "brain_subjects": SYN / "brain_subjects_list.csv",
    "brain_qc": SYN / "brain_qc.csv",
    "X_snps": SYN / "X_snps.npy",
    "y_snps": SYN / "y_snps.npy",
    "snp_samples": SYN / "snp_samples_list.csv",
    "metadata": SYN / "metadata.csv",
}

print("Synthetic dataset path:", SYN)
print("\nChecking presence of expected files:")
missing = []
for k, p in files_expected.items():
    ok = p.exists()
    print(f" - {p.name:30} : {'OK' if ok else 'MISSING'}")
    if not ok:
        missing.append(p)

if missing:
    print("\n>>> Missing files detected. Fix missing files before proceeding:")
    for m in missing:
        print("   -", m)
    # do not fail; continue to print what exists
print("\nLoading existing arrays (this may take a moment)...\n")

def load_if(p):
    if p.exists():
        try:
            if p.suffix == ".npy":
                return np.load(p)
            else:
                return pd.read_csv(p)
        except Exception as e:
            print(f"[ERROR] Failed to load {p}: {e}")
            return None
    return None

Xg = load_if(files_expected["X_genes"])
yg = load_if(files_expected["y_genes"])
gene_list = load_if(files_expected["gene_samples"])
Xb = load_if(files_expected["X_brain"])
yb = load_if(files_expected["y_brain"])
brain_list = load_if(files_expected["brain_subjects"])
qc = load_if(files_expected["brain_qc"])
Xs = load_if(files_expected["X_snps"])
ys = load_if(files_expected["y_snps"])
snp_list = load_if(files_expected["snp_samples"])
meta = load_if(files_expected["metadata"])

def shape_str(x):
    try:
        return str(x.shape)
    except Exception:
        return "n/a"

print("Shapes / basic info:")
if isinstance(Xg, np.ndarray):
    print(" - X_genes.npy shape:", Xg.shape)
if yg is not None:
    print(" - y_genes.npy shape:", np.array(yg).shape)
if isinstance(gene_list, pd.DataFrame):
    print(" - gene_samples_list rows:", gene_list.shape[0])

if isinstance(Xb, np.ndarray):
    print(" - X_brain.npy shape:", Xb.shape)
if yb is not None:
    print(" - y_brain.npy shape:", np.array(yb).shape)
if isinstance(brain_list, pd.DataFrame):
    print(" - brain_subjects_list rows:", brain_list.shape[0])

if isinstance(Xs, np.ndarray):
    print(" - X_snps.npy shape:", Xs.shape)
if ys is not None:
    print(" - y_snps.npy shape:", np.array(ys).shape)
if isinstance(snp_list, pd.DataFrame):
    print(" - snp_samples_list rows:", snp_list.shape[0])

if isinstance(meta, pd.DataFrame):
    print(" - metadata.csv rows:", meta.shape[0], "columns:", meta.shape[1])

# label balances
def label_stats(y, name):
    if y is None:
        print(f" - {name}: missing")
        return
    arr = np.array(y).ravel()
    try:
        unique, counts = np.unique(arr, return_counts=True)
        print(f" - {name} label distribution:")
        for u, c in zip(unique, counts):
            print(f"    {u} : {c}")
    except Exception as e:
        print(f" - {name}: cannot compute distribution ({e})")

label_stats(yg, "genes (y_genes)")
label_stats(yb, "brain (y_brain)")
label_stats(ys, "snps (y_snps)")

# sample id matching checks
def ids_from_df(df):
    if df is None: return None
    # try common column names
    for col in ("sample_id","subject_id","id","sample"):
        if col in df.columns:
            return df[col].astype(str).tolist()
    # fallback: first column
    return df.iloc[:,0].astype(str).tolist()

gids = ids_from_df(gene_list)
bids = ids_from_df(brain_list)
sids = ids_from_df(snp_list)
mid = None
if isinstance(meta, pd.DataFrame):
    # prioritize a sample id-like column
    for c in ("sample_id","subject_id","id"):
        if c in meta.columns:
            mid = meta[c].astype(str).tolist()
            break

print("\nSample ID checks:")
print(" - gene samples:", len(gids) if gids else "missing")
print(" - brain samples:", len(bids) if bids else "missing")
print(" - snp samples :", len(sids) if sids else "missing")
if gids and bids:
    common_gb = set(gids).intersection(set(bids))
    print(f" - genes ∩ brain : {len(common_gb)} (of gene {len(gids)} / brain {len(bids)})")
if gids and sids:
    print(" - genes ∩ snps :", len(set(gids).intersection(set(sids))))
if bids and sids:
    print(" - brain ∩ snps :", len(set(bids).intersection(set(sids))))

# quick numeric summaries
import numpy as _np
def quick_stats(arr, name):
    if arr is None: 
        print(f" - {name} missing")
        return
    a = np.asarray(arr, dtype=float)
    print(f" - {name} mean={_np.nanmean(a):.4g}, std={_np.nanstd(a):.4g}, min={_np.nanmin(a):.4g}, max={_np.nanmax(a):.4g}")

if isinstance(Xg, np.ndarray):
    quick_stats(Xg.mean(axis=1), "genes: per-sample mean")
if isinstance(Xb, np.ndarray):
    quick_stats(Xb.mean(axis=1), "brain: per-sample mean")
if isinstance(Xs, np.ndarray):
    quick_stats(Xs.mean(axis=1), "snps: per-sample mean (dosage)")

print("\nIf everything looks OK, proceed to training scripts. If sample ID sets do not match, you'll need to align them before fusion.")
print("Done.")

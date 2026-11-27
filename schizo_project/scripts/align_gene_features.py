#!/usr/bin/env python3
"""
Align X_genes.npy with metadata labels and produce consistent X / y / sample_list.

Writes (overwrites, after backup):
  data/processed/X_genes.npy
  data/processed/y_genes.npy
  data/processed/gene_samples_list.csv

It uses (in order of preference) metadata from:
  data/raw/genes/gene_sample_metadata.csv
  data/processed/genes/gene_sample_metadata_inferred.csv
  data/processed/genes/gene_sample_metadata_suggested.csv

Logic:
 - Load X and sample list (rows of X correspond to gene_samples_list.csv order)
 - Read metadata, find label column, normalize labels (control→0, schizophrenia→1)
 - Align by sample_id; keep only samples present in both X and metadata and with mappable labels
 - Backup original files to backups/align_genes_backup_<timestamp>/
 - Save filtered X, y, sample list
"""
from pathlib import Path
import numpy as np
import pandas as pd
import shutil
import datetime
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROC = PROJECT_ROOT / "data" / "processed"
RAW = PROJECT_ROOT / "data" / "raw" / "genes"

X_path = PROC / "X_genes.npy"
samples_path = PROC / "gene_samples_list.csv"

meta_candidates = [
    RAW / "gene_sample_metadata.csv",
    PROC / "gene_sample_metadata_inferred.csv",
    PROC / "gene_sample_metadata_suggested.csv",
]

# find metadata
meta_path = None
for p in meta_candidates:
    if p.exists():
        meta_path = p
        break

if meta_path is None:
    print("[ERROR] No metadata file found. Checked:")
    for p in meta_candidates:
        print("  -", p)
    sys.exit(2)

if not X_path.exists() or not samples_path.exists():
    print("[ERROR] Missing processed X or sample list:")
    print("  X:", X_path.exists(), X_path)
    print("  samples:", samples_path.exists(), samples_path)
    sys.exit(3)

print("[INFO] Using metadata:", meta_path)
print("[INFO] Loading X:", X_path)
X = np.load(X_path)
samples_df = pd.read_csv(samples_path, dtype=str)
# determine sample id column
if "sample_id" in samples_df.columns:
    sample_ids = samples_df["sample_id"].astype(str).tolist()
else:
    # assume first column
    sample_ids = samples_df.iloc[:, 0].astype(str).tolist()

if len(sample_ids) != X.shape[0]:
    print(f"[WARN] sample list length ({len(sample_ids)}) != X rows ({X.shape[0]}).")
    # Still proceed: assume X rows correspond in order to sample_ids if sample_ids length matches X rows, else create dummy ids.
    if len(sample_ids) != X.shape[0]:
        print("[ERROR] Cannot reliably align: sample_list length does not match X rows.")
        print("Please ensure data/processed/gene_samples_list.csv is correct.")
        sys.exit(4)

# load metadata
meta = pd.read_csv(meta_path, dtype=str)
meta_cols = [c.lower() for c in meta.columns]

# find sample id column in metadata
sid_col = None
for c in meta.columns:
    if c.lower() in ("sample_id","sample","geo_accession","gsm","id"):
        sid_col = c
        break
if sid_col is None:
    # fallback: first column
    sid_col = meta.columns[0]
    print(f"[WARN] Using first metadata column as sample id: {sid_col}")

meta = meta.set_index(sid_col, drop=False)
meta.index = meta.index.astype(str)

# find label column
label_col = None
for c in meta.columns:
    if c.lower() in ("label","diagnosis","inferred_label","group"):
        label_col = c
        break
if label_col is None:
    for c in meta.columns:
        if "label" in c.lower() or "diagnos" in c.lower():
            label_col = c
            break
if label_col is None:
    print("[ERROR] Could not find a label/diagnosis column in metadata. Columns:", list(meta.columns))
    sys.exit(5)

print("[INFO] Found label column:", label_col)

# normalize labels for all metadata rows
def map_label(v):
    if pd.isna(v): return None
    s = str(v).strip().lower()
    if s in ("control","healthy","normal","ctrl"): return 0
    if s in ("schizophrenia","schiz","sz"): return 1
    return None

meta["__label_bin__"] = meta[label_col].apply(map_label)

# align sample_ids (order of rows in X)
aligned_idxs = []
aligned_labels = []
dropped = []
for idx, sid in enumerate(sample_ids):
    sid_str = str(sid)
    if sid_str in meta.index:
        lbl = meta.loc[sid_str, "__label_bin__"]
        if lbl is not None:
            aligned_idxs.append(idx)
            aligned_labels.append(int(lbl))
        else:
            dropped.append((idx, sid_str, meta.loc[sid_str, label_col]))
    else:
        dropped.append((idx, sid_str, None))

print(f"[INFO] Samples in X: {len(sample_ids)}; will keep {len(aligned_idxs)} with valid binary labels; drop {len(dropped)}")

if len(aligned_idxs) == 0:
    print("[ERROR] No aligned samples with binary labels found. Aborting.")
    sys.exit(6)

# backup
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
backup_dir = PROJECT_ROOT / "backups" / f"align_genes_backup_{ts}"
backup_dir.mkdir(parents=True, exist_ok=True)
shutil.copy2(X_path, backup_dir / X_path.name)
shutil.copy2(samples_path, backup_dir / samples_path.name)
print("[INFO] Backed up X and sample list to:", backup_dir)

# create filtered arrays
keep_idx = aligned_idxs
X_filtered = X[keep_idx, :]
y_arr = np.array(aligned_labels, dtype=np.int8)
sample_ids_filtered = [sample_ids[i] for i in keep_idx]

# save
np.save(X_path, X_filtered)
np.save(PROC / "y_genes.npy", y_arr)
pd.DataFrame({"sample_id": sample_ids_filtered}).to_csv(samples_path, index=False)

print("[OK] Wrote aligned X_genes.npy (shape {}) and y_genes.npy (shape {})".format(X_filtered.shape, y_arr.shape))
if dropped:
    print("[WARN] Example dropped samples (index, id, raw_label):")
    for ex in dropped[:10]:
        print("  ", ex)
    print(f"[WARN] Total dropped: {len(dropped)}. Backup at: {backup_dir}")

print("Done.")

#!/usr/bin/env python3
"""
resolve_gene_labels.py

Create binary y_genes.npy from metadata, filtering out non-binary labels
(e.g. bipolar, major_depressive_disorder). Backs up original X and sample-list
before overwriting.

Usage:
    python scripts/resolve_gene_labels.py    # safe default: backup + overwrite

Outputs:
    data/processed/X_genes.npy              (OVERWRITTEN with filtered samples)
    data/processed/y_genes.npy              (binary 0/1 array)
    data/processed/gene_samples_list.csv   (filtered sample list)
    backups/processed_genes_backup/         (timestamped backups)
"""
from pathlib import Path
import pandas as pd
import numpy as np
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
    PROC / "gene_sample_metadata_template.csv",
]

# find metadata file (prefer canonical in raw/)
meta_path = None
for p in meta_candidates:
    if p.exists():
        meta_path = p
        break

if meta_path is None:
    print("[ERROR] No metadata file found (checked):")
    for p in meta_candidates:
        print("   -", p)
    sys.exit(2)

print("[INFO] Using metadata:", meta_path)

# load X and samples list
if not X_path.exists() or not samples_path.exists():
    print("[ERROR] Required processed files missing:")
    print("  -", X_path.exists(), X_path)
    print("  -", samples_path.exists(), samples_path)
    sys.exit(3)

X = np.load(X_path)  # shape: (n_samples, n_features) per your script
samples_df = pd.read_csv(samples_path)
if "sample_id" not in samples_df.columns and samples_df.shape[1] >= 1:
    # assume first column is sample id
    samples_df.columns = ["sample_id"] + list(samples_df.columns[1:])
sample_ids = samples_df["sample_id"].astype(str).tolist()
if len(sample_ids) != X.shape[0]:
    print(f"[ERROR] sample count mismatch: X rows={X.shape[0]} but sample list={len(sample_ids)}")
    sys.exit(4)

meta = pd.read_csv(meta_path, dtype=str)
meta_cols = [c.lower() for c in meta.columns]
if "sample_id" not in meta.columns and "sample" not in meta.columns and "geo_accession" not in meta.columns:
    # try to find candidate column name
    cand = None
    for c in meta.columns:
        if c.lower().startswith("sample") or c.lower().startswith("gsm") or "geo" in c.lower():
            cand = c
            break
    if cand:
        meta = meta.rename(columns={cand: "sample_id"})
    else:
        print("[ERROR] Could not locate sample id column in metadata. Columns found:", list(meta.columns))
        sys.exit(5)

# normalize column names for label search
label_col = None
for c in meta.columns:
    if c.lower() in ("label","diagnosis","inferred_label","group"):
        label_col = c
        break
# fallback heuristics
if label_col is None:
    for c in meta.columns:
        if "label" in c.lower() or "diagnos" in c.lower():
            label_col = c
            break

if label_col is None:
    print("[ERROR] Could not find label/diagnosis column in metadata. Columns:", list(meta.columns))
    sys.exit(6)

print("[INFO] Found label column:", label_col)

# prepare mapping
meta_map = meta.set_index(meta.columns[0])  # first col should be sample id
# ensure index is sample_id string matches samples list
meta_map.index = meta_map.index.astype(str)

# build aligned label list for sample_ids
labels_raw = []
missing_meta = []
for s in sample_ids:
    if s in meta_map.index:
        labels_raw.append(str(meta_map.loc[s, label_col]))
    else:
        labels_raw.append("")  # mark missing
        missing_meta.append(s)

if missing_meta:
    print(f"[WARN] {len(missing_meta)} samples missing metadata entries (will be dropped): {missing_meta[:10]}...")

# show label distribution
from collections import Counter
counter = Counter([str(l).strip().lower() for l in labels_raw if str(l).strip()])
print("[INFO] Label value counts (raw):")
for k,v in counter.most_common():
    print(f"  {k}: {v}")

# define allowed mapping
# map control/healthy/normal/ctrl -> 0, schizophrenia/sz -> 1
POSITIVE = ("schizophrenia","schiz", "sz")
NEGATIVE = ("control","healthy","normal","ctrl")

binary = []
keep_idx = []
dropped = []
for i,l in enumerate(labels_raw):
    s = str(l).strip().lower()
    if s in POSITIVE:
        binary.append(1)
        keep_idx.append(i)
    elif s in NEGATIVE:
        binary.append(0)
        keep_idx.append(i)
    else:
        # not mappable (bipolar, mdd, empty, other)
        dropped.append((i, sample_ids[i], s))
        # do not append value; we'll drop this sample

print(f"[INFO] To keep: {len(keep_idx)} samples. To drop (non-binary): {len(dropped)}")

if len(keep_idx) == 0:
    print("[ERROR] No samples map to binary labels. Aborting.")
    sys.exit(7)

# backup originals
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
backup_dir = PROJECT_ROOT / "backups" / f"processed_genes_backup_{ts}"
backup_dir.mkdir(parents=True, exist_ok=True)
shutil.copy2(X_path, backup_dir / X_path.name)
shutil.copy2(samples_path, backup_dir / samples_path.name)
print("[INFO] Backed up X and sample list to:", backup_dir)

# filter X and samples
X_filtered = X[keep_idx, :]
sample_ids_filtered = [sample_ids[i] for i in keep_idx]
y_arr = np.array([1 if str(labels_raw[i]).strip().lower() in POSITIVE else 0 for i in keep_idx], dtype=np.int8)

# overwrite processed outputs (filtered)
np.save(X_path, X_filtered)
np.save(PROC / "y_genes.npy", y_arr)
pd.DataFrame({"sample_id": sample_ids_filtered}).to_csv(samples_path, index=False)

print("[OK] Wrote filtered X_genes.npy (shape {}) and y_genes.npy (shape {})".format(X_filtered.shape, y_arr.shape))
if dropped:
    print("[WARN] Dropped samples (examples):")
    for i, sid, val in dropped[:10]:
        print(f"  {sid} -> '{val}'")
    print(f"Total dropped: {len(dropped)}. Originals backed up at: {backup_dir}")
print("Done.")

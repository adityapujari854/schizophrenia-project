#!/usr/bin/env python3
"""
validate_exp_grid.py

Simple validator for configs/exp_grid.csv:
 - ensures required columns exist
 - normalizes boolean columns (copy_all, overwrite)
 - prints summary and writes a sanitized copy to configs/exp_grid_clean.csv
"""
from pathlib import Path
import pandas as pd
import sys

ROOT = Path.cwd()
CFG = ROOT / "configs" / "exp_grid.csv"
OUT = ROOT / "configs" / "exp_grid_clean.csv"

if not CFG.exists():
    print(f"ERROR: {CFG} not found")
    sys.exit(1)

df = pd.read_csv(CFG, dtype=str).fillna("")
print(f"[INFO] Loaded {len(df)} rows from {CFG}")

# required columns (common ones used by experiment_runner)
required = ["out_dir","n_gene_signal","n_brain_signal","n_snp_signal","effect","fraction","seed","copy_all","overwrite"]
for c in required:
    if c not in df.columns:
        print(f"[WARN] Column '{c}' missing; adding default empty values")
        df[c] = ""

# normalize booleans
def norm_bool(x):
    s = str(x).strip().lower()
    return "True" if s in ("1","true","yes","t","y") else "False"

df['copy_all'] = df['copy_all'].apply(norm_bool)
df['overwrite'] = df['overwrite'].apply(norm_bool)

# write cleaned file
df.to_csv(OUT, index=False)
print(f"[OK] Wrote sanitized grid to {OUT}")
print(df.head(10).to_string(index=False))

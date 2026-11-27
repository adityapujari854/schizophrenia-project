#!/usr/bin/env python3
"""
clean_experiment_summary.py

Cleans results/experiment_summary.csv:
 - removes rows where out_dir is empty or equals "..."
 - deduplicates by out_dir keeping the latest timestamp
 - writes results/experiment_summary_clean.csv and backups original file
"""
from pathlib import Path
import pandas as pd
import shutil
import sys

ROOT = Path.cwd()
RESULTS = ROOT / "results"
SRC = RESULTS / "experiment_summary.csv"
OUT = RESULTS / "experiment_summary_clean.csv"
BACKUP = RESULTS / "experiment_summary_backup.csv"

if not SRC.exists():
    print(f"ERROR: {SRC} not found")
    sys.exit(1)

print(f"[INFO] Backing up original to {BACKUP}")
shutil.copy2(SRC, BACKUP)

df = pd.read_csv(SRC)
print(f"[INFO] Loaded {len(df)} rows")

# Normalize out_dir column to string, drop null/ellipsis
df['out_dir'] = df['out_dir'].astype(str).str.strip()
invalid = df['out_dir'].isin(['', '...', 'nan', 'None'])
if invalid.any():
    print(f"[INFO] Dropping {invalid.sum()} invalid rows (empty/out_dir placeholders)")
    df = df[~invalid].copy()

# Convert timestamp to numeric if possible
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')

# Deduplicate by out_dir keeping latest timestamp when present
if 'timestamp' in df.columns:
    df_sorted = df.sort_values('timestamp', ascending=False)
else:
    df_sorted = df
df_dedup = df_sorted.drop_duplicates(subset=['out_dir'], keep='first').reset_index(drop=True)
print(f"[INFO] Deduplicated -> {len(df_dedup)} rows")

# Write cleaned file
df_dedup.to_csv(OUT, index=False)
print(f"[OK] Wrote cleaned summary to {OUT}")

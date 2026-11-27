#!/usr/bin/env python3
"""
check_env.py

Quick environment and dataset sanity checks.
- verifies required Python packages can be imported
- checks project folder structure and presence of key raw files
- prints helpful instructions if something is missing
"""

import importlib
import sys
from pathlib import Path

REQUIRED_PKGS = [
    "numpy", "pandas", "scipy", "sklearn", "statsmodels",
    "matplotlib", "seaborn", "nibabel", "nilearn", "abagen", "openpyxl"
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
print("Project root:", PROJECT_ROOT)

# check python packages
print("\nChecking Python packages...")
missing = []
for pkg in REQUIRED_PKGS:
    try:
        importlib.import_module(pkg)
        print(f"  OK  {pkg}")
    except Exception:
        print(f"  MISSING  {pkg}")
        missing.append(pkg)

if missing:
    print("\n>>> Some packages are missing. Install them with:")
    print("    conda activate sz_brainml")
    print("    pip install " + " ".join(missing))
else:
    print("All required Python packages found.")

# check directory structure and required files
print("\nChecking expected files and directories...")
checks = [
    ("raw genes dir", PROJECT_ROOT / "data" / "raw" / "genes"),
    ("raw brain dir", PROJECT_ROOT / "data" / "raw" / "brain"),
    ("processed genes dir", PROJECT_ROOT / "data" / "processed" / "genes"),
    ("processed brain dir", PROJECT_ROOT / "data" / "processed" / "brain"),
    ("GSE series matrix", PROJECT_ROOT / "data" / "raw" / "genes" / "GSE53987_series_matrix.txt.gz"),
    ("GSE raw tar", PROJECT_ROOT / "data" / "raw" / "genes" / "GSE53987_RAW.tar"),
    ("gencode GTF/CSV", PROJECT_ROOT / "data" / "raw" / "genes" / "gencode.v19.chr_patch_hapl_scaff.annotation.gtf.gz")
]

for name, path in checks:
    if path.exists():
        print(f"  OK  {name}: {path.name}")
    else:
        print(f"  MISSING  {name}: expected at {path}")

# quick check that scripts dir exists
if (PROJECT_ROOT / "scripts").exists():
    print("\nScripts directory present.")
else:
    print("\nScripts directory missing at:", PROJECT_ROOT / "scripts")
    print("Create it and place the pipeline scripts there.")

print("\nQuick tips:")
print(" - If files are missing, place them under data/raw/... following the plan.")
print(" - If large datasets (OpenNeuro) are slow to download, use a mirror or download manager.")
print("\nDone.")
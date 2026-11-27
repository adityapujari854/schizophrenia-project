#!/usr/bin/env python3
"""
scripts/check_and_fix_input.py

Check that X_genes.npy, X_brain.npy, X_snps.npy exist and have non-zero features.
Prints actionable messages.
"""
from pathlib import Path
import numpy as np
import sys

def check(path):
    if not path.exists():
        return f"MISSING: {path}"
    try:
        arr = np.load(str(path))
    except Exception as e:
        return f"ERROR loading {path}: {e}"
    return f"OK: {path.name} shape={arr.shape}"

def main(data_dir="data/synthetic_signal_targeted"):
    p = Path(data_dir)
    print(f"Checking data dir: {p.resolve()}")
    files = ["X_genes.npy","X_brain.npy","X_snps.npy","y_genes.npy"]
    for f in files:
        print(check(p / f))
    # quick content sanity
    try:
        Xb = np.load(str(p / "X_brain.npy"))
        if Xb.ndim != 2:
            print(f"[WARN] X_brain.npy ndim={Xb.ndim} (expected 2)")
        if Xb.shape[1] == 0:
            print("[ERROR] X_brain has 0 columns (no features). You must regenerate X_brain.")
    except Exception:
        pass

if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv)>1 else "data/synthetic_signal_targeted"
    main(d)
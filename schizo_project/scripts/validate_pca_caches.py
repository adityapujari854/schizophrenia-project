#!/usr/bin/env python3
"""
validate_pca_caches.py
----------------------

Checks integrity of PCA cache files (results/pca_*.npz) used by patient_predict.py.

✅ Verifies that each .npz file:
   - Exists
   - Can be read
   - Contains 'X_pca' array
   - Has nonzero shape

Prints friendly colored output.
"""

import numpy as np
from pathlib import Path

def check_pca_cache(path: Path):
    if not path.exists():
        print(f"❌ Missing file: {path}")
        return False

    try:
        with np.load(str(path)) as data:
            if "X_pca" not in data:
                print(f"❌ Invalid format (missing 'X_pca'): {path}")
                return False

            X = data["X_pca"]
            if X.size == 0:
                print(f"❌ Empty array in {path}")
                return False

            print(f"✅ OK: {path.name} shape={X.shape}")
            return True

    except Exception as e:
        print(f"❌ Failed to read {path}: {e}")
        return False


def main():
    base = Path("results")
    print(f"Checking PCA caches under: {base.resolve()}\n")

    all_ok = True
    for tag in ["gene", "brain", "snp"]:
        path = base / f"pca_{tag}.npz"
        print(f"Checking {path}...")
        ok = check_pca_cache(path)
        all_ok = all_ok and ok
        print()

    if all_ok:
        print("🎯 All PCA caches look good!")
    else:
        print("⚠️  Some PCA caches are missing or corrupted. Recreate them using:")
        print("   python scripts/reduce_genes_to_pca.py --data-dir data/synthetic_fullrun --out-dir results --n-components 200 --save-full")
        print("   python scripts/reduce_brain_to_pca.py --data-dir data/synthetic_fullrun --out-dir results --n-components 500 --save-full")
        print("   python scripts/reduce_snps_to_pca.py --data-dir data/synthetic_fullrun --out-dir results --n-components 50 --save-full")

if __name__ == "__main__":
    main()

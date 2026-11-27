# scripts/validate_grid.py
from pathlib import Path
import csv, sys

p = Path("configs/exp_grid.csv")
if not p.exists():
    print("configs/exp_grid.csv not found")
    sys.exit(1)

bad = []
with open(p, newline='', encoding='utf-8') as fh:
    r = csv.DictReader(fh)
    for i, row in enumerate(r, start=1):
        od = (row.get("out_dir") or row.get("outdir") or "").strip()
        if not od or od in (".", "..", "..."):
            bad.append((i, od))
if bad:
    print("Found invalid out_dir entries on rows:")
    for i, od in bad:
        print(f" row {i}: '{od}'")
    sys.exit(2)
print("Looks OK.")

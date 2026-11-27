#!/usr/bin/env python3
"""
Aggregate per-subject *_connectivity_z.npy into X_brain.npy with handling of mismatched ROI sizes.

Usage:
  python scripts/aggregate_brain_subjects.py --in-dir data/processed/brain/subjects --out-dir data/processed/brain
  python scripts/aggregate_brain_subjects.py --in-dir data/processed/brain/subjects --out-dir data/processed/brain --allow-pad --pad-with nan

By default the script selects the most common ROI-count and skips others.
Use --allow-pad to pad smaller vectors to the largest vector length (pad with NaN or zero).
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import sys

def flatten_upper(mat: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(mat.shape[0], k=1)
    return mat[iu]

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", type=Path, required=True, help="Directory with per-subject *_connectivity_z.npy")
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory to write X_brain.npy and brain_subjects_list.csv")
    p.add_argument("--allow-pad", action="store_true", help="If set, pad smaller feature vectors to the largest length (default: skip mismatched)")
    p.add_argument("--pad-with", choices=["nan","zero"], default="nan", help="Pad with NaN (default) or zero when --allow-pad is used")
    args = p.parse_args(argv)

    indir = args.in_dir
    outdir = args.out_dir
    outdir.mkdir(parents=True, exist_ok=True)

    files = sorted(indir.glob("*_connectivity_z.npy"))
    if not files:
        print("[ERROR] No *_connectivity_z.npy files found in", indir)
        return 1

    info = []
    groups = defaultdict(list)
    skipped = []
    for f in files:
        subj = f.name.split("_")[0]
        try:
            mat = np.load(f)
        except Exception as e:
            print(f"[WARN] Could not load {f.name}: {e}; skipping")
            skipped.append((subj, f.name, "load_error"))
            continue
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            print(f"[WARN] {f.name} is not square (shape={mat.shape}); skipping")
            skipped.append((subj, f.name, f"bad_shape:{mat.shape}"))
            continue
        n = mat.shape[0]
        vec = flatten_upper(mat)
        groups[n].append((subj, vec, f.name))
        info.append({"subject": subj, "file": f.name, "n_rois": n, "vec_len": vec.size})

    info_df = pd.DataFrame(info)
    info_df.to_csv(outdir / "connectivity_files_info.csv", index=False)
    counts = Counter([r["n_rois"] for r in info])
    print("[INFO] ROI-size counts:", dict(counts))

    if len(groups) == 1:
        # all same size -> straightforward stack
        n = next(iter(groups))
        rows = [v for (_, v, _) in groups[n]]
        X = np.vstack(rows)
        subj_ids = [s for (s, _, _) in groups[n]]
    else:
        # multiple ROI sizes present
        # choose most common size by default
        most_common_n, mc_count = counts.most_common(1)[0]
        print(f"[WARN] Multiple ROI sizes found. Most common: {most_common_n} ({mc_count} subjects).")
        if args.allow_pad:
            # pad all vectors to max length
            max_n = max(groups.keys())
            max_len = max((vec.size for grp in groups.values() for (_, vec, _) in grp))
            pad_with = np.nan if args.pad_with == "nan" else 0.0
            print(f"[INFO] Padding all feature vectors to length {max_len} (pad_with={args.pad_with})")
            rows = []
            subj_ids = []
            for n, items in groups.items():
                for subj, vec, fname in items:
                    if vec.size < max_len:
                        pad = np.full((max_len - vec.size,), pad_with, dtype=float)
                        vec_p = np.concatenate([vec.astype(float), pad])
                    else:
                        vec_p = vec.astype(float)
                    rows.append(vec_p)
                    subj_ids.append(subj)
            X = np.vstack(rows)
        else:
            # keep only most common size
            kept = groups[most_common_n]
            rows = [v for (_, v) in kept]
            subj_ids = [s for (s, _, _) in kept]
            X = np.vstack(rows)
            # log skipped entries (all not in kept)
            for n, items in groups.items():
                if n == most_common_n:
                    continue
                for subj, vec, fname in items:
                    skipped.append((subj, fname, f"mismatch_n_{n}"))
            print(f"[INFO] Kept {len(subj_ids)} subjects with ROI-count={most_common_n}; skipped {len(skipped)} mismatched files.")
    # Save outputs
    np.save(outdir / "X_brain.npy", X)
    pd.DataFrame({"subject_id": subj_ids}).to_csv(outdir / "brain_subjects_list.csv", index=False)
    # save info and skipped report
    if skipped:
        skip_df = pd.DataFrame(skipped, columns=["subject","file","reason"])
        skip_df.to_csv(outdir / "mismatch_report.csv", index=False)
        print(f"[INFO] Wrote mismatch_report.csv with {len(skipped)} entries")
    print(f"[OK] Saved X_brain.npy (shape={X.shape}) and brain_subjects_list.csv in {outdir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
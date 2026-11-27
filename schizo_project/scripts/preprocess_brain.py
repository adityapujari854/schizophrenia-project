#!/usr/bin/env python3
"""
scripts/preprocess_brain.py

Full brain preprocessing for ds000030 subset (BIDS-structured folders expected).

Features:
 - Scans data/raw/brain/<dataset>/ for subject sub-<id>/func/ subfolders
 - For each subject:
    * find BOLD nifti (func) and confounds TSV (BIDS confounds)
    * extract ROI timeseries using Nilearn NiftiLabelsMasker with a Schaefer atlas (200 parcels)
    * clean timeseries with nilearn.signal.clean (bandpass + confounds regression + detrend + standardize)
    * compute Pearson correlation matrix, Fisher z-transform
    * compute motion QC (mean FD) if confounds include framewise_displacement or compute FD from motion params
 - Save per-subject:
    data/processed/brain/<sub>_timeseries.npy
    data/processed/brain/<sub>_connectivity_z.npy
 - Save aggregated:
    data/processed/X_brain.npy         (subjects x features [upper-triangle flattened])
    data/processed/brain_subjects_list.csv
    data/processed/brain_qc.csv

Usage:
    # dry run (no heavy computation)
    python scripts/preprocess_brain.py --bids-root data/raw/brain/ds000030 --atlas-path /path/to/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz --dry-run

    # actual run (example atlas will be auto-downloaded if not provided? -- see notes)
    python scripts/preprocess_brain.py --bids-root data/raw/brain/ds000030 --out-dir data/processed/brain

Notes:
 - Requires: nibabel, nilearn, numpy, pandas. Install in your env:
       pip install nibabel nilearn numpy pandas
 - Atlas: by default the script will try to use Nilearn's fetch_atlas_schaefer_2018 if available.
   You may pass a local atlas NIfTI via --atlas-path.
 - Use --exclude-high-motion to exclude subjects with mean FD > threshold from final X_brain.npy (but still report in QC).
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Import neuro libs
try:
    import nibabel as nib
    from nilearn.maskers import NiftiLabelsMasker
    from nilearn import signal, datasets
except Exception as e:
    raise RuntimeError("nibabel and nilearn are required. Install via: pip install nibabel nilearn") from e


# -----------------------
# Helper functions
# -----------------------
def find_subject_fmri_files(bids_root: Path) -> List[Tuple[str, Path, Optional[Path]]]:
    """
    Search BIDS-like structure under bids_root for subject folders and find:
      - functional NIfTI (file whose name contains '_bold' or endswith '.nii'/.nii.gz)
      - confounds TSV (file name contains 'confounds' or 'desc-confounds_regressors.tsv')
    Returns list of tuples: (sub_id, func_path, confounds_path_or_None)
    """
    subs = []
    if not bids_root.exists():
        raise FileNotFoundError(f"BIDS root not found: {bids_root}")
    # assume structure: bids_root/sub-*/func/*.nii(.gz) and confounds.tsv
    for subdir in sorted(bids_root.glob("sub-*")):
        func_dir = subdir / "func"
        if not func_dir.exists():
            continue
        # prefer files with task or _bold in name
        func_candidates = [p for p in func_dir.glob("*.nii*") if ("bold" in p.name or "task" in p.name or "func" in p.name)]
        if not func_candidates:
            # fallback any nifti
            func_candidates = list(func_dir.glob("*.nii*"))
        if not func_candidates:
            continue
        func_path = sorted(func_candidates)[0]  # pick first if multiple
        # find confounds file
        conf_candidates = [p for p in func_dir.glob("*confounds*.tsv")] + [p for p in func_dir.glob("*desc-confounds_regressors.tsv")]
        conf_path = sorted(conf_candidates)[0] if conf_candidates else None
        subs.append((subdir.name, func_path, conf_path))
    return subs


def read_confounds(conf_path: Path) -> pd.DataFrame:
    """Read confounds TSV into DataFrame (if present)."""
    if conf_path is None or not conf_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(conf_path, sep="\t")
    return df


def select_confounds_for_regression(conf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pick motion and aCompCor confounds typically available:
      - framewise_displacement, trans_x/y/z, rot_x/y/z, a_comp_cor_*
    Returns DataFrame with selected confounds (or empty DF).
    """
    if conf_df is None or conf_df.empty:
        return pd.DataFrame()
    cols = []
    for c in ["framewise_displacement", "framewise-displacement", "fd"]:
        if c in conf_df.columns:
            cols.append(c)
            break
    # motion params
    motion_cols = [c for c in conf_df.columns if any(x in c.lower() for x in ("trans", "rot", "motion", "tx", "ty", "tz", "rx", "ry", "rz"))]
    # aCompCor
    acomp = [c for c in conf_df.columns if c.lower().startswith("a_comp_cor")]
    # Combine unique, preserving order
    for c in motion_cols + acomp:
        if c not in cols:
            cols.append(c)
    sel = [c for c in cols if c in conf_df.columns]
    return conf_df[sel] if sel else pd.DataFrame()


def compute_fd_stats(conf_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute FD stats: mean FD, pct frames > 0.5 mm
    If FD column exists, use it. Otherwise try to compute FD from motion params (approx).
    """
    if conf_df is None or conf_df.empty:
        return {"mean_FD": float("nan"), "pct_FD_gt_0.5": float("nan"), "n_vols": 0}
    fd_col = None
    for c in conf_df.columns:
        if "framewise" in c.lower() or c.lower() == "fd":
            fd_col = c
            break
    if fd_col:
        fd = pd.to_numeric(conf_df[fd_col], errors="coerce")
        mean_fd = float(fd.mean(skipna=True))
        pct = float((fd > 0.5).sum() / max(1, fd.shape[0]))
        return {"mean_FD": mean_fd, "pct_FD_gt_0.5": pct, "n_vols": int(fd.shape[0])}
    # fallback: attempt to compute FD from motion params columns (translations+rotations)
    mp_cols = [c for c in conf_df.columns if any(x in c.lower() for x in ("trans", "rot", "tx", "ty", "tz", "rx", "ry", "rz"))]
    if len(mp_cols) >= 6:
        mp = conf_df[mp_cols].astype(float).fillna(0.0)
        # simple framewise difference sum of absolute differences (translations+rotations)
        diff = mp.diff().abs().sum(axis=1).fillna(0.0)
        mean_fd = float(diff.mean())
        pct = float((diff > 0.5).sum() / max(1, diff.shape[0]))
        return {"mean_FD": mean_fd, "pct_FD_gt_0.5": pct, "n_vols": int(mp.shape[0])}
    # otherwise unknown
    return {"mean_FD": float("nan"), "pct_FD_gt_0.5": float("nan"), "n_vols": 0}


def fisher_z_transform(corr: np.ndarray) -> np.ndarray:
    """Fisher z transform on correlation matrix. Keep diagonal as zeros."""
    # clip to avoid inf
    corr = np.clip(corr, -0.999999, 0.999999)
    # replace NaNs (e.g., constant signals) with 0
    corr = np.nan_to_num(corr, nan=0.0)
    z = np.arctanh(corr)
    np.fill_diagonal(z, 0.0)
    return z


def upper_triangle_flatten(mat: np.ndarray) -> np.ndarray:
    """Return flattened upper-triangle (excluding diagonal) of a square matrix as 1D vector."""
    idx = np.triu_indices(mat.shape[0], k=1)
    return mat[idx]


# -----------------------
# Main processing
# -----------------------
def process_subject(sub_id: str, func_niifile: Path, confound_tsv: Optional[Path], masker: NiftiLabelsMasker,
                    tr: Optional[float], low_pass: Optional[float], high_pass: Optional[float],
                    clean_kwargs: Dict) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Process a single subject:
     - load func nifti
     - load confounds and select columns
     - extract ROI timeseries with masker
     - clean (regress confounds, bandpass, detrend, standardize)
     - compute correlation matrix and fisher-z
    Returns (timeseries ndarray: n_regions x n_vols, zcorr ndarray: n_regions x n_regions, qc_dict)
    """
    img = nib.load(str(func_niifile))

    # try to get TR from header (if provided)
    hdr = img.header
    if tr is None:
        zooms = hdr.get_zooms()
        tr_est = float(zooms[3]) if (len(zooms) >= 4) else None
    else:
        tr_est = tr

    conf_df = read_confounds(confound_tsv) if confound_tsv else pd.DataFrame()
    conf_for_reg = select_confounds_for_regression(conf_df)
    qc = compute_fd_stats(conf_df)

    # extract timeseries: masker returns (n_timepoints, n_regions)
    try:
        ts_time_regions = masker.fit_transform(img)
    except Exception as e:
        raise RuntimeError(f"Masker fit_transform failed for {sub_id}: {e}")
    if ts_time_regions.ndim != 2:
        raise RuntimeError(f"Unexpected timeseries shape from masker for {sub_id}: {ts_time_regions.shape}")
    # transpose to (n_regions, n_timepoints)
    ts = ts_time_regions.T

    # Clean timeseries using nilearn.signal.clean (expects shape (n_timepoints, n_signals))
    try:
        ts_clean_time_regions = signal.clean(ts_time_regions,
                                             confounds=(conf_for_reg.values if (conf_for_reg is not None and not conf_for_reg.empty) else None),
                                             t_r=tr_est,
                                             low_pass=low_pass,
                                             high_pass=high_pass,
                                             detrend=clean_kwargs.get("detrend", True),
                                             standardize=clean_kwargs.get("standardize", True))
    except Exception as e:
        raise RuntimeError(f"nilearn.signal.clean failed for {sub_id}: {e}")
    # convert to (n_regions, n_timepoints)
    ts_clean = ts_clean_time_regions.T

    # check for constant signals -> avoid NaNs in correlation
    if np.any(np.std(ts_clean, axis=1) == 0):
        # replace constant rows with small noise to avoid zero-division
        const_idx = np.where(np.std(ts_clean, axis=1) == 0)[0]
        for ci in const_idx:
            ts_clean[ci] = ts_clean[ci] + 1e-6 * np.random.RandomState(0).randn(ts_clean.shape[1])

    # compute correlation (regions x regions)
    corr = np.corrcoef(ts_clean)
    z = fisher_z_transform(corr)

    # update QC: n_vols and tr
    qc.update({"n_vols": int(ts_clean.shape[1]), "TR": tr_est if tr_est is not None else float("nan")})
    return ts_clean, z, qc


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="preprocess_brain.py",
                                     description="Preprocess BOLD NIfTIs to extract ROI timeseries and connectivity.")
    parser.add_argument("--bids-root", type=Path, required=True, help="BIDS-like root folder for dataset (e.g. data/raw/brain/ds000030)")
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/brain"), help="Output directory")
    parser.add_argument("--atlas-path", type=Path, default=None, help="Path to atlas NIfTI labels (optional). If omitted, fetch Schaefer atlas via nilearn.datasets.fetch_atlas_schaefer_2018")
    parser.add_argument("--n-parcels", type=int, default=200, help="Number of parcels in Schaefer atlas when fetching (default 200)")
    parser.add_argument("--bandpass", type=str, default="0.01-0.1", help="Bandpass freq low-high in Hz (e.g. 0.01-0.1) or 'none' to skip")
    parser.add_argument("--exclude-high-motion", action="store_true", help="Exclude high motion subjects (mean FD > threshold) from final X_brain.npy")
    parser.add_argument("--fd-thresh", type=float, default=0.5, help="Framewise displacement threshold (mm) for exclusion")
    parser.add_argument("--dry-run", action="store_true", help="Dry run: list subjects and show planned actions")
    args = parser.parse_args(argv)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    per_sub_dir = out_dir / "subjects"
    per_sub_dir.mkdir(parents=True, exist_ok=True)

    # atlas handling
    atlas_img = None
    if args.atlas_path:
        atlas_img = args.atlas_path
        if not atlas_img.exists():
            print(f"[ERROR] Provided atlas not found: {atlas_img}")
            return 2
    else:
        # fetch Schaefer atlas from nilearn (if available)
        print("[INFO] Fetching Schaefer atlas via nilearn.datasets.fetch_atlas_schaefer_2018 ...")
        try:
            sch = datasets.fetch_atlas_schaefer_2018(n_rois=args.n_parcels, yeo_networks=7, verbose=1)
            atlas_img = Path(sch.maps)
        except Exception as e:
            print("[ERROR] Could not fetch Schaefer atlas automatically:", e)
            print("Please provide an atlas NIfTI via --atlas-path")
            return 3

    # masker setup
    masker = NiftiLabelsMasker(labels_img=str(atlas_img), standardize=False, memory="nilearn_cache")

    # parse bandpass args (format "low-high")
    if args.bandpass.lower() == "none":
        low_pass = None
        high_pass = None
    else:
        try:
            low_s, high_s = args.bandpass.split("-")
            low = float(low_s)
            high = float(high_s)
            # nilearn.signal.clean expects low_pass (upper freq cutoff) and high_pass (lower cutoff)
            low_pass = high
            high_pass = low
        except Exception:
            print("[WARN] Could not parse bandpass argument; using default 0.01-0.1")
            low_pass = 0.1
            high_pass = 0.01

    # find subjects
    subjects = find_subject_fmri_files(args.bids_root)
    if not subjects:
        print("[ERROR] No subjects/fMRI files found under bids root:", args.bids_root)
        return 4

    print(f"[INFO] Found {len(subjects)} subjects to process (dry-run={args.dry_run})")

    results_rows = []
    subject_timeseries = {}
    subject_connectivity = {}
    for sub_id, func_path, conf_path in subjects:
        print(f"[INFO] Subject {sub_id}: func={func_path.name} confounds={'present' if conf_path else 'none'}")
        if args.dry_run:
            continue
        try:
            ts, zmat, qc = process_subject(sub_id, func_path, conf_path, masker,
                                           tr=None, low_pass=low_pass, high_pass=high_pass,
                                           clean_kwargs={"detrend": True, "standardize": True})
        except Exception as e:
            print(f"[ERROR] Failed processing {sub_id}: {e}")
            continue

        # Save per-subject timeseries and connectivity
        ts_out = per_sub_dir / f"{sub_id}_timeseries.npy"
        conn_out = per_sub_dir / f"{sub_id}_connectivity_z.npy"
        np.save(ts_out, ts)
        np.save(conn_out, zmat)
        print(f"[OK] Saved {ts_out.name} (shape {ts.shape}), {conn_out.name} (shape {zmat.shape})")

        # flatten upper triangle
        feat_vec = upper_triangle_flatten(zmat)
        subject_timeseries[sub_id] = feat_vec
        subject_connectivity[sub_id] = zmat
        qc_row = {"subject_id": sub_id}
        qc_row.update(qc)
        results_rows.append(qc_row)

    if args.dry_run:
        print("[DRYRUN] subjects detected:")
        for sub_id, func_path, conf_path in subjects:
            print(" -", sub_id, func_path.name, "confounds:", bool(conf_path))
        print("[DRYRUN] No files written.")
        return 0

    # Assemble X_brain.npy (subjects x features)
    subj_ids = list(subject_timeseries.keys())
    if not subj_ids:
        print("[ERROR] No subject processed successfully.")
        return 5
    feature_matrix = np.vstack([subject_timeseries[sid] for sid in subj_ids])
    # QC df
    qc_df = pd.DataFrame(results_rows)
    # optionally exclude high-motion
    if args.exclude_high_motion:
        keep_mask = qc_df["mean_FD"].fillna(0) <= args.fd_thresh
        keep_subs = qc_df.loc[keep_mask, "subject_id"].tolist()
        filtered_idx = [i for i, sid in enumerate(subj_ids) if sid in keep_subs]
        feature_matrix = feature_matrix[filtered_idx, :]
        subj_ids = [sid for sid in subj_ids if sid in keep_subs]
        print(f"[INFO] Excluding high-motion subjects: kept {len(subj_ids)} subjects after FD thresh {args.fd_thresh} mm")

    # Save aggregated outputs
    np.save(out_dir / "X_brain.npy", feature_matrix)
    pd.DataFrame({"subject_id": subj_ids}).to_csv(out_dir / "brain_subjects_list.csv", index=False)
    qc_df.to_csv(out_dir / "brain_qc.csv", index=False)
    print(f"[OK] Saved X_brain.npy -> {out_dir / 'X_brain.npy'} (shape: {feature_matrix.shape})")
    print(f"[OK] Saved brain_subjects_list.csv and brain_qc.csv -> {out_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""
scripts/convert_cel_to_csv.py

Convert Affymetrix CEL(.gz/.CEL) files to per-sample CSVs of probe intensities,
and optionally a combined probe x sample CSV.

Usage examples:
    # single file -> per-sample CSV
    python scripts/convert_cel_to_csv.py --input data/raw/genes/GSE53987_RAW/GSM1305056.CEL.gz

    # directory -> parse all .CEL.gz and .CEL files, write combined table
    python scripts/convert_cel_to_csv.py --input-dir data/raw/genes/GSE53987_RAW --combined

    # show what would happen (no files written)
    python scripts/convert_cel_to_csv.py --input-dir data/raw/genes/GSE53987_RAW --dry-run

Notes:
 - For robust binary CEL parsing, install Biopython (which provides Bio.Affy).
   Without it the script tries a best-effort text parsing fallback which may fail for many CEL variants.
"""
from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Try to import Bio.Affy (Biopython). If not available, we'll fallback.
try:
    from Bio.Affy import CelFile  # type: ignore
    _HAS_BIOAFFY = True
except Exception:
    _HAS_BIOAFFY = False

# -----------------------
# Helpers
# -----------------------
def read_cel_with_bioaffy(path: Path) -> Tuple[List[str], np.ndarray]:
    """
    Read CEL using Bio.Affy.CelFile.read(handle) and return (probe_ids_list, intensities_array_or_1d).
    - If CelFile object provides probe_ids/probes and intensities, match them.
    - If only a 2D grid is present, flatten and create probe ids 'x_y'.
    """
    # open binary (Bio.Affy expects binary handle)
    with (path.open("rb")) as fh:
        cel = CelFile.read(fh)

    # prefer explicit probe ids if present
    probe_ids = None
    if hasattr(cel, "probes") and cel.probes is not None:
        probe_ids = [str(p) for p in cel.probes]
    elif hasattr(cel, "probe_ids") and cel.probe_ids is not None:
        probe_ids = [str(p) for p in cel.probe_ids]
    # intensities attribute (may be 1d or 2d)
    intensities = getattr(cel, "intensities", None)
    if intensities is None:
        # Some CelFile variants expose signal or vals; attempt common alternatives
        possible = []
        for attr in ("signal", "intensity", "value", "values"):
            if hasattr(cel, attr):
                possible.append(getattr(cel, attr))
        if possible:
            intensities = possible[0]
    if intensities is None:
        raise RuntimeError(f"No intensity-like attribute found in CelFile for {path}")

    arr = np.asarray(intensities)
    if arr.ndim == 1:
        # 1D intensities; if we have probe_ids of same length, good
        if probe_ids is not None and len(probe_ids) == arr.size:
            return probe_ids, arr.astype(float)
        else:
            # create generic probe ids
            ids = [f"p{i}" for i in range(arr.size)]
            return ids, arr.astype(float)
    elif arr.ndim == 2:
        # 2D grid: if we have matching probe_ids with same length, prefer that;
        # otherwise create x_y coordinate ids after flattening in row-major order.
        flat = arr.reshape(-1)
        if probe_ids is not None and len(probe_ids) == flat.size:
            return probe_ids, flat.astype(float)
        else:
            coords = []
            rows, cols = arr.shape
            for i in range(rows):
                for j in range(cols):
                    coords.append(f"{i}_{j}")
            return coords, flat.astype(float)
    else:
        raise RuntimeError(f"Unsupported intensity array shape {arr.shape} in {path}")


def read_cel_text_fallback(path: Path) -> Tuple[List[str], List[float]]:
    """
    A best-effort text fallback parser for CEL-like text files.
    It searches for lines containing a numeric intensity value and treats the first token as probe id.
    This will not work for many binary CELs; prefer Biopython.
    """
    text = ""
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt", errors="ignore") as fh:
            text = fh.read()
    else:
        text = path.read_text(errors="ignore")

    # gather candidate lines that look like: <probe> <...> <intensity>
    candidates = []
    for ln in text.splitlines():
        if not ln.strip():
            continue
        parts = ln.strip().split()
        # try last token as float
        try:
            val = float(parts[-1])
            candidates.append((parts[0], val))
        except Exception:
            continue
    if not candidates:
        raise RuntimeError("No numeric candidate lines found in CEL text fallback parsing.")
    probes, vals = zip(*candidates)
    return list(probes), list(map(float, vals))


def parse_single_cel(path: Path) -> Tuple[str, pd.Series]:
    """
    Parse a single CEL file and return (sample_name, series) where series index = probe_id and values = intensities.
    sample_name is taken from filename (without extensions).
    """
    sample_name = path.name.split(".")[0]
    if _HAS_BIOAFFY:
        probe_ids, intens = read_cel_with_bioaffy(path)
        ser = pd.Series(data=intens, index=probe_ids, name=sample_name)
        return sample_name, ser
    else:
        # fallback
        probe_ids, vals = read_cel_text_fallback(path)
        ser = pd.Series(data=vals, index=probe_ids, name=sample_name)
        return sample_name, ser


# -----------------------
# CLI / main
# -----------------------
def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="convert_cel_to_csv.py",
                                description="Convert Affymetrix CEL(.gz/.CEL) files to CSV probe-intensity tables.")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=Path, help="Single CEL(.gz/.CEL) file to convert")
    group.add_argument("--input-dir", type=Path, help="Directory containing CEL files (use --pattern to change glob)")
    p.add_argument("--pattern", type=str, default="*.CEL.gz", help="Glob pattern when using --input-dir (default: *.CEL.gz)")
    p.add_argument("--out-dir", type=Path, default=Path("data/processed/genes/cel_converted"), help="Output directory")
    p.add_argument("--combined", action="store_true", help="Also write combined_probe_table.csv (probes x samples) if multiple samples parsed")
    p.add_argument("--dry-run", action="store_true", help="Do not write files; just print planned actions")
    args = p.parse_args(argv)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # assemble list of CELs
    cel_paths: List[Path] = []
    if args.input:
        if not args.input.exists():
            print(f"[ERROR] Input file not found: {args.input}")
            return 2
        cel_paths = [args.input]
    else:
        if not args.input_dir or not args.input_dir.exists():
            print(f"[ERROR] Input directory not found: {args.input_dir}")
            return 3
        # support several common extensions if pattern yields nothing
        cel_paths = sorted(list(args.input_dir.glob(args.pattern)))
        if not cel_paths:
            # try common alternates
            alt = sorted(list(args.input_dir.glob("*.CEL")) + list(args.input_dir.glob("*.cel.gz")) + list(args.input_dir.glob("*.CEL.gz")))
            cel_paths = alt

    if not cel_paths:
        print("[ERROR] No CEL files found to parse.")
        return 4

    print(f"[INFO] Found {len(cel_paths)} CEL file(s) to parse.")

    # parse each file and save per-sample CSV
    samples_series: Dict[str, pd.Series] = {}
    for pth in cel_paths:
        print(f"[INFO] Parsing: {pth}")
        try:
            sample_name, series = parse_single_cel(pth)
        except Exception as e:
            print(f"[ERROR] Failed to parse {pth}: {e}")
            continue

        # sanitize index: ensure unique probe ids; if duplicates exist, aggregate by mean
        if not series.index.is_unique:
            print(f"[WARN] Duplicate probe IDs detected for sample {sample_name}; aggregating by mean.")
            series = series.groupby(series.index).mean()

        per_out = out_dir / f"{sample_name}_probe_intensities.csv"
        if args.dry_run:
            print(f"[DRYRUN] Would write per-sample CSV: {per_out} (rows: {series.shape[0]})")
        else:
            # write as two-column CSV: probe_id,intensity
            df_out = pd.DataFrame({ "probe_id": series.index.astype(str), sample_name: series.values })
            df_out.to_csv(per_out, index=False)
            print(f"[OK] Wrote {per_out} (rows: {df_out.shape[0]})")

        samples_series[sample_name] = series

    # optionally build combined probe x sample table
    if args.combined and len(samples_series) >= 1:
        combined_out = out_dir / "combined_probe_table.csv"
        if args.dry_run:
            print(f"[DRYRUN] Would create combined table with {len(samples_series)} samples (probes union).")
        else:
            # concat aligning on index (probe ids). This will produce NaNs for missing probes in some samples.
            combined_df = pd.concat(samples_series.values(), axis=1, join="outer")
            # reorder columns to stable sample order
            combined_df = combined_df.loc[:, list(samples_series.keys())]
            combined_df.to_csv(combined_out, index=True)  # index are probe IDs
            print(f"[OK] Wrote combined probe table -> {combined_out} (shape: {combined_df.shape})")

    if not _HAS_BIOAFFY:
        print("[NOTE] Biopython (Bio.Affy) not available — used text fallback which may be unreliable for binary CELs.")
        print("Install biopython to improve parsing: pip install biopython")

    print("[DONE] convert_cel_to_csv finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

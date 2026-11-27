#!/usr/bin/env python3
"""
scripts/data_download.py

Selects a balanced subset of ds000030 (v1.0.0) subjects (SCHZ + CONTROL),
estimates download size (anat + rest fMRI + confounds per subject),
and downloads subjects until total estimated size <= MAX_BYTES (10GB by default).

Place this file at schizo_project/scripts/data_download.py and run:
    conda activate sz_brainml
    python scripts/data_download.py

Requirements: requests, pandas, tqdm
"""
from __future__ import annotations

import sys
import time
import urllib.parse
from pathlib import Path
from typing import List, Tuple

import requests
import pandas as pd
from tqdm import tqdm

# ---------- CONFIG ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

PARTICIPANTS_PATH = PROJECT_ROOT / "data" / "raw" / "brain" / "participants.tsv"
OUT_BRAIN_DIR = PROJECT_ROOT / "data" / "raw" / "brain" / "ds000030"

# OpenNeuro snapshot v1.0.0 files endpoint
# Note: we construct file URLs by percent-encoding path components
OPENNEURO_BASE = "https://openneuro.org/crn/datasets/ds000030/snapshots/1.0.0/files"

# Files per subject we want (relative path templates)
REL_FILES = [
    "{sub}/anat/{sub}_T1w.nii.gz",
    "{sub}/func/{sub}_task-rest_bold.nii.gz",
    # confounds may not be present in raw dataset; keep as attempt
    "{sub}/func/{sub}_task-rest_bold_confounds.tsv",
]

# maximum bytes to download (default 10 GB)
MAX_BYTES = 10 * 1024 ** 3

# HTTP timeout for HEAD requests (seconds)
HEAD_TIMEOUT = 15

# chunk size for streaming download
CHUNK_SIZE = 1024 * 1024  # 1 MB


# ---------- Helpers ----------
def openneuro_file_url(relative_path: str) -> str:
    """
    Construct an OpenNeuro file URL with percent-encoding for each path
    component. We append ?download=1 to get the raw file content.
    """
    rp = relative_path.lstrip("/")
    parts = [urllib.parse.quote(p) for p in rp.split("/")]
    rp_enc = "/".join(parts)
    return f"{OPENNEURO_BASE}/{rp_enc}?download=1"


def human_readable(n: int) -> str:
    """Return human readable file size"""
    n = float(n)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024.0:
            return f"{n:3.2f}{unit}"
        n /= 1024.0
    return f"{n:.2f}PB"


def get_remote_size(url: str) -> int:
    """
    Return content-length in bytes if available, else 0.
    Uses HEAD first; if not available, tries GET with Range 0-0.
    """
    try:
        r = requests.head(url, allow_redirects=True, timeout=HEAD_TIMEOUT)
        if r.status_code >= 400:
            # some servers may not allow HEAD; try GET range
            r = requests.get(url, headers={"Range": "bytes=0-0"}, stream=True, timeout=HEAD_TIMEOUT)
        r.raise_for_status()
        cl = r.headers.get("Content-Length") or r.headers.get("content-length")
        if cl:
            return int(cl)
    except Exception:
        # any failure yields unknown size (0)
        return 0
    return 0


def download_file_resumable(url: str, dest: Path) -> bool:
    """
    Resumable download using Range if supported.
    Writes to dest + ".part" temporary file while downloading and renames on completion.
    Returns True on success (or if file already exists), False otherwise.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")

    # fetch remote content length if possible
    remote_size = 0
    try:
        head = requests.head(url, allow_redirects=True, timeout=HEAD_TIMEOUT)
        if head.status_code >= 400:
            # try GET 0-0
            head = requests.get(url, headers={"Range": "bytes=0-0"}, stream=True, timeout=HEAD_TIMEOUT)
        head.raise_for_status()
        remote_size = int(head.headers.get("Content-Length", 0) or 0)
    except Exception:
        remote_size = 0

    # if dest already is complete, skip
    if dest.exists() and remote_size and dest.stat().st_size == remote_size:
        print(f"[SKIP] {dest.name} already exists ({human_readable(dest.stat().st_size)})")
        return True

    # determine resume offset
    existing = part.stat().st_size if part.exists() else 0
    headers = {}
    if existing and remote_size and existing < remote_size:
        headers["Range"] = f"bytes={existing}-"

    try:
        with requests.get(url, stream=True, headers=headers, allow_redirects=True, timeout=HEAD_TIMEOUT) as r:
            # if server responded with 416 Range Not Satisfiable, restart full download
            if r.status_code == 416:
                existing = 0
                if part.exists():
                    part.unlink()
                r = requests.get(url, stream=True, allow_redirects=True, timeout=HEAD_TIMEOUT)

            r.raise_for_status()
            mode = "ab" if ("Range" in headers and existing) else "wb"
            total = remote_size or int(r.headers.get("Content-Length", 0) or 0)

            downloaded = existing
            start = time.time()
            with open(part, mode) as fh:
                # iterate over chunks and write
                for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    downloaded += len(chunk)
                    # print progress every ~0.5s
                    elapsed = time.time() - start
                    if elapsed > 0.5:
                        if total:
                            pct = downloaded / total * 100
                            sys.stdout.write(f"\r  {dest.name}: {human_readable(downloaded)}/{human_readable(total)} ({pct:.1f}%)")
                        else:
                            sys.stdout.write(f"\r  {dest.name}: {human_readable(downloaded)}")
                        sys.stdout.flush()
            # final newline after progress
            print()
    except Exception as e:
        print(f"[ERROR] Download failed for {url}: {e}")
        return False

    # rename .part -> final dest
    try:
        part.replace(dest)
    except Exception as e:
        # fallback to rename
        try:
            part.rename(dest)
        except Exception as e2:
            print(f"[WARN] Could not finalize download file rename: {e2}")
    # report
    try:
        print(f"[OK] Saved {dest} ({human_readable(dest.stat().st_size)})")
    except Exception:
        print(f"[OK] Saved {dest} (size unknown)")
    return True


# ---------- Selection logic ----------
def pick_subjects_by_size(participants_path: Path, max_bytes: int = MAX_BYTES):
    """
    Read participants.tsv, filter those with rest & T1w available (best-effort),
    classify into CONTROL/SCHZ/OTHER and select subjects alternating control/schz
    until max_bytes exceeded (based on HEAD-estimated sizes).
    Returns list of tuples: (subject_id, list_of_file_info)
    where file_info = (relative_path, url, estimated_size_bytes)
    """
    if not participants_path.exists():
        raise FileNotFoundError(f"participants.tsv not found at {participants_path}")

    df = pd.read_csv(participants_path, sep="\t", dtype=str)
    df.columns = [c.strip() for c in df.columns]

    # find participant column if named differently
    if "participant_id" not in df.columns:
        possible = [c for c in df.columns if c.lower().startswith("sub") or c.lower().startswith("participant")]
        if possible:
            df = df.rename(columns={possible[0]: "participant_id"})
        else:
            raise ValueError("participant_id column not found in participants.tsv")

    # ensure columns exist
    for c in ["rest", "T1w", "diagnosis"]:
        if c not in df.columns:
            df[c] = None

    # helper to parse various truthy markers
    def has_flag(x):
        try:
            return bool(float(x))
        except Exception:
            try:
                return str(x).strip().lower() in ("1", "true", "yes")
            except Exception:
                return False

    df["rest_ok"] = df["rest"].apply(has_flag)
    df["T1w_ok"] = df["T1w"].apply(has_flag)
    df = df[(df["rest_ok"]) & (df["T1w_ok"])]

    df["diag_norm"] = df["diagnosis"].astype(str).fillna("").str.strip().str.upper()

    def group_label(s: str) -> str:
        if "SCH" in s:
            return "SCHZ"
        if "CONT" in s:
            return "CONTROL"
        return "OTHER"

    df["group"] = df["diag_norm"].apply(group_label)

    controls = df[df["group"] == "CONTROL"]["participant_id"].tolist()
    schz = df[df["group"] == "SCHZ"]["participant_id"].tolist()
    others = df[~df["participant_id"].isin(controls + schz)]["participant_id"].tolist()

    selection: List[Tuple[str, List[Tuple[str, str, int]]]] = []
    cumulative = 0
    idx_c = idx_s = idx_o = 0

    def try_add(sub: str):
        nonlocal cumulative
        estimated = 0
        sizes = []
        for tmpl in REL_FILES:
            rel = tmpl.format(sub=sub)
            url = openneuro_file_url(rel)
            sz = get_remote_size(url)
            sizes.append((rel, url, sz))
            estimated += sz
        # if estimated would exceed cap and we know sizes, reject
        if cumulative + estimated > max_bytes and estimated > 0:
            return False, sizes
        cumulative += estimated
        selection.append((sub, sizes))
        return True, sizes

    # alternate adding control / schz until no progress
    while True:
        progressed = False
        if idx_c < len(controls):
            sub = controls[idx_c]
            ok, _ = try_add(sub)
            idx_c += 1
            if ok:
                progressed = True
        if idx_s < len(schz):
            sub = schz[idx_s]
            ok, _ = try_add(sub)
            idx_s += 1
            if ok:
                progressed = True
        if idx_o < len(others) and not progressed:
            sub = others[idx_o]
            ok, _ = try_add(sub)
            idx_o += 1
            if ok:
                progressed = True
        if not progressed:
            break

    # Print summary
    print("\nSelection summary (based on HEAD-estimated sizes):")
    total_known = 0
    total_unknown_count = 0
    for sub, sizes in selection:
        print(f"  Subject: {sub}")
        for rel, url, sz in sizes:
            print(f"    {rel} -> {human_readable(sz) if sz else 'unknown'}")
            if sz:
                total_known += sz
            else:
                total_unknown_count += 1

    print(f"\nEstimated known total size: {human_readable(total_known)}")
    if total_unknown_count:
        print(f"Files with unknown size: {total_unknown_count} (these will be downloaded but counted as unknown)")

    print(f"Download cap: {human_readable(max_bytes)}")
    print(f"Estimated cumulative known size after selection: {human_readable(total_known)}")
    return selection


# ---------- Main ----------
def main():
    print("Starting subject selection based on participants.tsv and size cap.")
    try:
        selection = pick_subjects_by_size(PARTICIPANTS_PATH, max_bytes=MAX_BYTES)
    except Exception as exc:
        print("[ERROR] Selection failed:", exc)
        sys.exit(1)

    if not selection:
        print("No subjects selected (maybe none with rest+T1w or cap too small). Exiting.")
        return

    OUT_BRAIN_DIR.mkdir(parents=True, exist_ok=True)

    overall_bytes = 0
    for sub, filesizes in selection:
        print(f"\nDownloading subject: {sub}")
        for rel, url, sz in filesizes:
            dest = OUT_BRAIN_DIR / rel
            try:
                ok = download_file_resumable(url, dest)
                if ok and dest.exists():
                    overall_bytes += dest.stat().st_size
            except Exception as e:
                print(f"  [ERROR] {e}")

    print("\nAll downloads attempted.")
    print(f"Total bytes actually downloaded (sum of finished files): {human_readable(overall_bytes)}")
    print(f"Files are in: {OUT_BRAIN_DIR}")
    print("If any .part files remain, re-run the script to resume them.")


if __name__ == "__main__":
    main()

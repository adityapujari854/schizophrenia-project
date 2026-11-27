#!/usr/bin/env python3
"""
scripts/download_ds000030_via_nilearn.py

Robust downloader for a small subset of OpenNeuro ds000030 using nilearn's file index.

What it does:
 - uses nilearn.datasets.fetch_ds000030_urls() to obtain a dataset index (URLs)
 - for each subject in SUBJECTS, finds exact URL(s) for:
     - func/{sub}_task-rest_bold.nii.gz
     - func/{sub}_task-rest_bold.json
 - downloads each target file with retries, reports progress, and saves to:
     data/raw/brain/ds000030/<sub>/func/

Usage:
    pip install nilearn requests tqdm
    python scripts/download_ds000030_via_nilearn.py

Notes:
 - This avoids guessing file URLs and uses the same index Nilearn uses.
 - If the dataset snapshot changes, nilearn will fetch the appropriate index version.
"""

from pathlib import Path
import time
import sys
import requests
from tqdm import tqdm

# User picks (quick balanced set)
SUBJECTS = ["sub-10159", "sub-50004", "sub-10171", "sub-50005"]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "data" / "raw" / "brain" / "ds000030"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# target relative file patterns inside the dataset index
TARGET_PATTERNS = [
    "/{sub}/func/{sub}_task-rest_bold.nii.gz",
    "/{sub}/func/{sub}_task-rest_bold.json",
]

# download config
TIMEOUT = 30
CHUNK = 1024 * 1024
MAX_RETRIES = 3
SLEEP_BETWEEN_RETRIES = 2.0  # seconds


def fetch_index():
    """Use nilearn to fetch the ds000030 index of file URLs."""
    try:
        from nilearn.datasets import fetch_ds000030_urls
    except Exception as e:
        raise RuntimeError("nilearn is required: pip install nilearn") from e

    print("[INFO] Downloading ds000030 file index (small)...")
    result = fetch_ds000030_urls(verbose=1)
    # fetch_ds000030_urls returns (urls_path, urls) or similar depending on Nilearn version.
    # It may return a tuple; handle both.
    if isinstance(result, tuple):
        # often (path, urls)
        urls = result[1]
    else:
        urls = result
    if not urls:
        raise RuntimeError("Failed to fetch ds000030 urls from nilearn.")
    print(f"[INFO] Index contains {len(urls)} entries.")
    return urls


def find_url_for_pattern(urls, pattern):
    """Return the first URL from the index that endswith the given pattern (or contains it)."""
    # pattern is something like '/sub-10159/func/sub-10159_task-rest_bold.nii.gz'
    candidates = [u for u in urls if pattern in u]
    if candidates:
        # prefer exact match at end
        for u in candidates:
            if u.endswith(pattern):
                return u
        return candidates[0]
    return None


def download_url(url: str, dest: Path) -> bool:
    """Download url to dest with streaming and retries."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    # skip existing large files
    if dest.exists() and dest.stat().st_size > 1024 * 1024:
        print(f"[SKIP] {dest} exists ({dest.stat().st_size} bytes)")
        return True

    attempt = 0
    while attempt < MAX_RETRIES:
        attempt += 1
        try:
            with requests.get(url, stream=True, timeout=TIMEOUT) as r:
                if r.status_code == 404:
                    print(f"[MISSING] {url} -> 404")
                    return False
                r.raise_for_status()
                total = int(r.headers.get("Content-Length") or 0)
                tmp = dest.with_suffix(dest.suffix + ".part")
                downloaded = 0
                with open(tmp, "wb") as fh, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as pbar:
                    for chunk in r.iter_content(chunk_size=CHUNK):
                        if not chunk:
                            continue
                        fh.write(chunk)
                        downloaded += len(chunk)
                        pbar.update(len(chunk))
                tmp.replace(dest)
                print(f"[OK] {dest} ({dest.stat().st_size} bytes)")
                return True
        except Exception as e:
            print(f"[WARN] Attempt {attempt} failed for {url}: {e}")
            time.sleep(SLEEP_BETWEEN_RETRIES)
    print(f"[ERROR] All {MAX_RETRIES} attempts failed for: {url}")
    return False


def main():
    urls = fetch_index()

    any_fail = False
    for sub in SUBJECTS:
        for pattern_template in TARGET_PATTERNS:
            pattern = pattern_template.format(sub=sub)
            url = find_url_for_pattern(urls, pattern)
            if not url:
                print(f"[WARN] No index entry found for pattern {pattern} (subject {sub})")
                any_fail = True
                continue
            # construct destination path inside OUT_DIR preserving dataset relative dirs after leading '/'
            # e.g. if url contains '/sub-10159/func/...', keep that suffix
            # find the index of '/sub-' occurrence in url
            idx = url.find("/sub-")
            rel = url[idx + 1 :] if idx != -1 else pattern.lstrip("/")
            dest = OUT_DIR / Path(rel)
            ok = download_url(url, dest)
            if not ok:
                any_fail = True

    if any_fail:
        print("\nFinished with some missing files or errors. Check messages above.")
    else:
        print("\nAll requested files downloaded successfully.")
    print("Files saved under:", OUT_DIR)
    print("Now run:")
    print("  python scripts/preprocess_brain.py --bids-root data/raw/brain/ds000030 --out-dir data/processed/brain")


if __name__ == "__main__":
    main()

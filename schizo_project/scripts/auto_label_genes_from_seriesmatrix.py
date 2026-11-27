#!/usr/bin/env python3
"""
scripts/auto_label_genes_from_seriesmatrix.py

Reads a GEO series matrix (GSE53987 by default), extracts per-sample metadata
and attempts to auto-infer sample labels (e.g., schizophrenia vs control).

Outputs (default into data/processed/genes/):
  - gene_sample_metadata_suggested.csv   (all parsed metadata + inferred_label)
  - gene_sample_metadata_inferred.csv    (sample_id,label)  [only inferred labels]
  - gene_sample_metadata_template.csv    (sample_id,label)  [samples needing manual label]

Usage:
    # default run (auto-detects gz or txt)
    python scripts/auto_label_genes_from_seriesmatrix.py

    # specify paths and dry-run
    python scripts/auto_label_genes_from_seriesmatrix.py \
        --series data/raw/genes/GSE53987_series_matrix.txt.gz \
        --out data/processed/genes --dry-run

Author: improved for Aditya Pujari (Nimbus project)
"""
from __future__ import annotations

import argparse
import gzip
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# -----------------------
# Defaults & configuration
# -----------------------
PROJECT_ROOT = Path.cwd()
DEFAULT_RAW_GENES = PROJECT_ROOT / "data" / "raw" / "genes"
DEFAULT_PROC_GENES = PROJECT_ROOT / "data" / "processed" / "genes"
DEFAULT_SERIES_CANDIDATES = [
    DEFAULT_RAW_GENES / "GSE53987_series_matrix.txt.gz",
    DEFAULT_RAW_GENES / "GSE53987_series_matrix.txt",
]

# Keywords used for label inference (extend as needed)
LABEL_KEYWORDS = {
    "schizophrenia": [r"\bschizophrenia\b", r"\bschizo\w*\b", r"\bsz\b"],
    "control": [r"\bcontrol\b", r"\bhealthy\b", r"\bnormal\b", r"\bctrl\b"],
    "bipolar": [r"\bbipolar\b", r"\bbd\b"],
    "major_depressive_disorder": [r"\bmajor depressive\b", r"\bmdd\b", r"\bdepress\b"],
}

# characteristic separators to split "characteristics_ch1" blobs
CHAR_SPLIT_RE = re.compile(r"\s*\|\s*|\s*;\s*|\s*,\s*|\s*/\s*")

# regexes
GSM_RE = re.compile(r"GSM\d+", re.IGNORECASE)
CHAR_KV_RE = re.compile(r"^\s*([^:;=]+)\s*[:=]\s*(.+)$")  # matches "age: 52" or "gender=M"

# -----------------------
# Utility functions
# -----------------------
def open_text(path: Path):
    """Open text file, support .gz."""
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", errors="replace")
    return open(path, "r", errors="replace")


def read_series_lines(path: Path) -> List[str]:
    with open_text(path) as fh:
        return [ln.rstrip("\n") for ln in fh]


def detect_series_file(candidates: List[Path]) -> Path:
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"No series matrix found in candidates: {candidates}")


def detect_sample_order(lines: List[str]) -> List[str]:
    """
    Try multiple strategies to determine sample order:
      1. !Sample_geo_accession line (preferred)
      2. header after !series_matrix_table_begin
      3. extract unique GSM tokens as fallback
    """
    # 1) !Sample_geo_accession
    for ln in lines:
        if ln.startswith("!Sample_geo_accession"):
            parts = ln.split("\t")
            return [clean_val(p) for p in parts[1:]]

    # 2) header after table begin
    for i, ln in enumerate(lines):
        if ln.strip().startswith("!series_matrix_table_begin") and i + 1 < len(lines):
            hdr = lines[i + 1].split("\t")
            if len(hdr) > 1:
                return [clean_val(p) for p in hdr[1:]]

    # 3) collect GSM tokens
    seen = []
    for ln in lines:
        found = GSM_RE.findall(ln)
        for g in found:
            if g not in seen:
                seen.append(g)
    return seen


def clean_val(v: str) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    # strip surrounding quotes
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    return s


def normalize_meta_key(raw_key: str) -> str:
    k = raw_key.lstrip("!").lstrip("^").strip()
    if k.lower().startswith("sample_"):
        k = k[len("sample_") :]
    return k.strip().lower()


def collect_meta_lines(lines: List[str]) -> List[str]:
    """Return lines starting with ! or ^ that contain a tab (meta lines)."""
    return [ln for ln in lines if (ln.startswith("!") or ln.startswith("^")) and ("\t" in ln)]


def parse_meta_lines(meta_lines: List[str], sample_order: List[str]) -> Dict[str, Dict[str, str]]:
    """
    Parse meta lines into dict: {sample_id: {key: value, ...}}
    - Repeated keys are concatenated with " | "
    """
    samples = {s: {} for s in sample_order}
    for ln in meta_lines:
        parts = ln.split("\t")
        if len(parts) < 2:
            continue
        raw_key = parts[0]
        key = normalize_meta_key(raw_key)
        vals = [clean_val(v) for v in parts[1:]]
        # ensure length
        if len(vals) < len(sample_order):
            vals += [""] * (len(sample_order) - len(vals))
        elif len(vals) > len(sample_order):
            vals = vals[: len(sample_order)]
        for sid, v in zip(sample_order, vals):
            if v == "":
                continue
            prev = samples[sid].get(key, "")
            samples[sid][key] = (prev + " | " + v) if (prev and v) else v
    return samples


def parse_characteristics(text: str) -> Dict[str, str]:
    """
    Parse a characteristics string into subfields.
    - Looks for key:value pairs first.
    - Otherwise stores raw tokens under 'value' or heuristics for region/diagnosis.
    """
    out = {}
    if not text:
        return out
    parts = CHAR_SPLIT_RE.split(text)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        m = CHAR_KV_RE.match(part)
        if m:
            key = m.group(1).strip().lower().replace(" ", "_")
            val = m.group(2).strip()
            out[key] = val
        else:
            # heuristic: check for 'region, diagnosis' or 'diagnosis, region'
            if "," in part:
                a, b = [p.strip() for p in part.split(",", 1)]
                # if one side matches disorder words consider it diagnosis
                if re.search(r"schiz|bipol|depress|control|healthy|mdd", b, re.I):
                    out.setdefault("brain_region", a)
                    out.setdefault("diagnosis_guess", b)
                elif re.search(r"schiz|bipol|depress|control|healthy|mdd", a, re.I):
                    out.setdefault("brain_region", b)
                    out.setdefault("diagnosis_guess", a)
                else:
                    out.setdefault("value", (out.get("value", "") + " | " + part) if out.get("value") else part)
            else:
                # store token as generic value
                out.setdefault("value", (out.get("value", "") + " | " + part) if out.get("value") else part)
    return out


def infer_label_from_text(text: str) -> str:
    """
    Infer label using LABEL_KEYWORDS and a few heuristics (Dx:, Diagnosis:, case/control tokens)
    Returns empty string if no confident inference.
    """
    if not text:
        return ""
    t = text.lower()

    # 1) direct keywords
    for label, patterns in LABEL_KEYWORDS.items():
        for patt in patterns:
            if re.search(patt, t, re.I):
                return label

    # 2) look for Dx: or diagnosis:
    m = re.search(r"(?:dx|diagnos(?:is)?)[:=]\s*([a-zA-Z0-9 _-]+)", t)
    if m:
        lab = m.group(1).strip().lower()
        for label, patterns in LABEL_KEYWORDS.items():
            for patt in patterns:
                if re.search(patt, lab, re.I) or (patt.strip("\\b") in lab):
                    return label

    # 3) case/patient vs control heuristics
    if re.search(r"\b(case|patient|affected)\b", t) and not re.search(r"\b(control|healthy|unaffected)\b", t):
        return "schizophrenia"
    if re.search(r"\b(control|healthy|unaffected)\b", t) and not re.search(r"\b(case|patient|affected)\b", t):
        return "control"

    # 4) region + disorder combos heuristics
    if re.search(r"hippocampus.*schiz|pre[- ]?frontal.*schiz|ba46.*schiz", t):
        return "schizophrenia"

    return ""


# -----------------------
# Main processing pipeline
# -----------------------
def build_metadata_dataframe(samples_meta: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    """
    Convert samples_meta dict to a pandas DataFrame with:
      - sample_id, title, source_name, description, raw_characteristics, combined_text, inferred_label
      - parsed characteristics as separate 'char_<key>' columns
    """
    rows = []
    for sid, meta in samples_meta.items():
        title = meta.get("title", "")
        source_name = meta.get("source_name_ch1", "") or meta.get("source_name", "")
        description = meta.get("description", "")
        raw_chars = meta.get("characteristics_ch1", "") or meta.get("characteristics", "")

        # parse characteristics into small dict
        parsed = parse_characteristics(raw_chars)
        # also consider other characteristic-like keys (sometimes multiple fields exist)
        # gather any key in meta that contains 'characteristics'
        for k, v in meta.items():
            if k.startswith("characteristics") and k != "characteristics_ch1":
                # merge parsed results from other char fields
                parsed.update(parse_characteristics(v))

        # build combined_text for inference (use title, source, desc, raw_chars, and parsed kvs)
        combined_parts = [title, source_name, description, raw_chars]
        for pk, pv in parsed.items():
            combined_parts.append(f"{pk}:{pv}")
        combined_text = " | ".join([p for p in combined_parts if p])

        inferred = infer_label_from_text(combined_text)

        row = {
            "sample_id": sid,
            "title": title,
            "source_name": source_name,
            "description": description,
            "raw_characteristics": raw_chars,
            "combined_text": combined_text,
            "inferred_label": inferred,
        }
        # add parsed characteristics as explicit columns with prefix char_
        for pk, pv in parsed.items():
            row[f"char_{pk}"] = pv

        # add any other top-level meta fields that are short and useful
        for opt in ("organism_ch1", "organism", "geo_accession", "sample_type"):
            if opt in meta and meta[opt]:
                row[opt] = meta[opt]

        rows.append(row)

    df = pd.DataFrame(rows)
    # reorder columns: sample_id first, combined_text just before inferred_label
    cols = list(df.columns)
    if "sample_id" in cols:
        cols.remove("sample_id")
        cols = ["sample_id"] + cols
    if "inferred_label" in cols:
        cols.remove("inferred_label")
        # place combined_text before inferred_label (if present)
        if "combined_text" in cols:
            cols = [c for c in cols if c != "combined_text"] + ["combined_text", "inferred_label"]
        else:
            cols = cols + ["inferred_label"]
    # ensure we only use valid columns
    final_cols = [c for c in cols if c in df.columns]
    df = df[final_cols]
    return df


def run(series_path: Path, out_dir: Path, dry_run: bool = False) -> Tuple[Path, Path, Path]:
    """
    Run the full extraction and saving. Returns paths of (suggested, inferred, template) CSVs.
    """
    lines = read_series_lines(series_path)
    sample_order = detect_sample_order(lines)
    if not sample_order:
        raise RuntimeError("Could not determine sample order in series matrix.")

    meta_lines = collect_meta_lines(lines)
    samples_meta = parse_meta_lines(meta_lines, sample_order)

    df = build_metadata_dataframe(samples_meta)

    out_dir.mkdir(parents=True, exist_ok=True)
    suggested_path = out_dir / "gene_sample_metadata_suggested.csv"
    inferred_path = out_dir / "gene_sample_metadata_inferred.csv"
    template_path = out_dir / "gene_sample_metadata_template.csv"

    if dry_run:
        # quick preview
        print("[DRY RUN] Would write suggested ->", suggested_path)
        print("[DRY RUN] Would write inferred ->", inferred_path)
        print("[DRY RUN] Would write template ->", template_path)
        print("\n[DRY RUN] Preview (first 5 rows):")
        print(df.head().to_string(index=False))
        return suggested_path, inferred_path, template_path

    # save full suggested metadata
    df.to_csv(suggested_path, index=False)

    # inferred-only (samples with a non-empty inferred_label)
    inferred_df = df[df["inferred_label"].astype(str).str.strip() != ""].copy()
    if not inferred_df.empty:
        inferred_df[["sample_id", "inferred_label"]].rename(columns={"inferred_label": "label"}).to_csv(inferred_path, index=False)
    else:
        # write empty template with headers for downstream clarity
        pd.DataFrame(columns=["sample_id", "label"]).to_csv(inferred_path, index=False)

    # template for manual labeling (samples without inferred label)
    template_df = df[df["inferred_label"].astype(str).str.strip() == ""][["sample_id"]].copy()
    template_df["label"] = ""
    template_df.to_csv(template_path, index=False)

    return suggested_path, inferred_path, template_path


# -----------------------
# CLI / entrypoint
# -----------------------
def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="auto_label_genes_from_seriesmatrix.py",
                                description="Extract sample metadata and infer labels from GEO series matrix.")
    p.add_argument("--series", type=Path, default=None,
                   help="Path to series matrix (.txt or .txt.gz). If omitted auto-detects common locations.")
    p.add_argument("--out", type=Path, default=DEFAULT_PROC_GENES,
                   help="Directory to write metadata CSVs (default: data/processed/genes)")
    p.add_argument("--dry-run", action="store_true", help="Do not write files; print a small preview.")
    args = p.parse_args(argv)

    try:
        series_path = args.series or detect_series_file(DEFAULT_SERIES_CANDIDATES)
    except FileNotFoundError as e:
        print("[ERROR]", e)
        return 2

    print("[INFO] Using series matrix:", series_path)
    try:
        suggested, inferred, template = run(series_path, args.out, dry_run=args.dry_run)
    except Exception as e:
        print("[ERROR] Failed to parse series matrix:", e)
        return 3

    if args.dry_run:
        print("[INFO] Dry-run finished.")
        return 0

    # Summary print
    full = pd.read_csv(suggested)
    inferred_df = pd.read_csv(inferred)
    template_df = pd.read_csv(template)
    print(f"[OK] Parsed {len(full)} samples.")
    print(f"[OK] Auto-inferred labels: {len(inferred_df)} -> {Path(inferred).name}")
    print(f"[OK] Need manual labeling (template): {len(template_df)} -> {Path(template).name}")
    print(f"[OK] Full suggested metadata -> {Path(suggested).name}")
    if not inferred_df.empty:
        print("\nInferred label counts:")
        print(inferred_df["label"].value_counts(dropna=False).to_string())
    print("\nNext steps:")
    print(f" - Review {inferred} and {suggested}")
    print(f" - If correct, copy {inferred} -> data/raw/genes/gene_sample_metadata.csv")
    print(f" - Otherwise, fill {template} and save to data/raw/genes/gene_sample_metadata.csv")
    print(" - After finalizing metadata, run: python scripts/gene_feature_assembly.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
scripts/dataset.py

Sanity loader / dataset scanner.

Checks:
 - presence of gene expression processed file (CSV) and optional gene metadata (sample -> label)
 - presence of BIDS participants.tsv (optional)
 - reports shapes, sample counts, unmatched samples, label counts
 - writes results/mismatched_samples.csv when mismatches found

Exit codes:
 - 0: OK
 - 2: Missing critical files
 - 3: Sample mismatch (requires manual fix)

PowerShell usage example:
  python .\scripts\dataset.py --genes-file data\processed\genes\GSE53987_processed_topvar.csv --gene-meta data\raw\genes\gene_sample_metadata.csv --participants data\raw\brain\ds000030\participants.tsv
"""
from pathlib import Path
import argparse
import sys
import pandas as pd
import numpy as np

def parse_args():
    p = argparse.ArgumentParser(description="Sanity-check dataset files and sample alignment")
    p.add_argument("--genes-file", type=str, default="data/processed/genes/GSE53987_processed_topvar.csv",
                   help="Processed gene expression CSV (rows=features, cols=samples)")
    p.add_argument("--gene-meta", type=str, default="data/raw/genes/gene_sample_metadata.csv",
                   help="(Optional) gene sample metadata CSV with columns sample_id,label")
    p.add_argument("--participants", type=str, default="data/raw/brain/ds000030/participants.tsv",
                   help="(Optional) BIDS participants.tsv for brain data")
    p.add_argument("--out-dir", type=str, default="results",
                   help="Directory to write mismatch CSVs and short reports")
    return p.parse_args()

def read_genes(path: Path):
    # read CSV with index as first column (features), columns are sample IDs
    df = pd.read_csv(path, index_col=0, header=0)
    return df

def read_gene_meta(path: Path):
    df = pd.read_csv(path, dtype=str)
    # normalize column names
    cols = [c.strip() for c in df.columns]
    df.columns = cols
    # try to find sample_id column
    if "sample_id" not in df.columns:
        # try variants
        for c in df.columns:
            if c.lower().startswith("sample") or c.lower().startswith("sub") or c.lower().startswith("participant"):
                df = df.rename(columns={c: "sample_id"})
                break
    if "sample_id" not in df.columns:
        raise ValueError("gene_meta provided but no sample id column found (expected 'sample_id')")
    return df

def read_participants(path: Path):
    # participants.tsv typical BIDS tab-separated file
    df = pd.read_csv(path, sep="\t", dtype=str)
    # normalize column names
    df.columns = [c.strip() for c in df.columns]
    return df

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def main():
    args = parse_args()
    genes_path = Path(args.genes_file)
    gene_meta_path = Path(args.gene_meta)
    participants_path = Path(args.participants)
    out_dir = Path(args.out_dir)
    safe_mkdir(out_dir)

    missing = []
    if not genes_path.exists():
        missing.append(str(genes_path))
    # gene_meta & participants are optional
    if missing:
        print("[ERROR] Missing required files:", "; ".join(missing))
        sys.exit(2)

    # Load genes
    try:
        genes_df = read_genes(genes_path)
    except Exception as e:
        print("[ERROR] Failed to read genes file:", e)
        sys.exit(2)

    print(f"[INFO] Loaded gene matrix: {genes_path} -> shape (features, samples) = {genes_df.shape}")
    # samples are columns
    gene_sample_ids = list(genes_df.columns)
    n_samples = len(gene_sample_ids)
    n_features = genes_df.shape[0]

    # Try to load gene metadata
    meta_present = False
    gene_meta_df = None
    if gene_meta_path.exists():
        try:
            gene_meta_df = read_gene_meta(gene_meta_path)
            meta_present = True
            # ensure sample_id and label columns
            if "label" not in gene_meta_df.columns:
                # try to find label-like column
                label_col = None
                for c in gene_meta_df.columns:
                    if c.lower() in ("label", "diagnosis", "group", "class"):
                        label_col = c
                        break
                if label_col:
                    gene_meta_df = gene_meta_df.rename(columns={label_col: "label"})
                else:
                    # keep it but warn
                    print("[WARN] gene metadata found but no 'label' column detected.")
            # display distribution if available
            if "label" in gene_meta_df.columns:
                # normalize label strings
                vc = gene_meta_df["label"].astype(str).str.strip().value_counts()
                print("[INFO] Gene metadata label counts (raw):")
                for lab, cnt in vc.items():
                    print(f"  {lab}: {cnt}")
        except Exception as e:
            print("[WARN] Could not read gene metadata:", e)
            meta_present = False

    else:
        print("[INFO] No gene metadata found at", gene_meta_path)

    # Try to load participants.tsv if present
    participants_present = False
    parts_df = None
    if participants_path.exists():
        try:
            parts_df = read_participants(participants_path)
            participants_present = True
            # try to find participant_id and diagnosis columns
            pid_cols = [c for c in parts_df.columns if c.lower() in ("participant_id", "participant", "sub")]
            diag_cols = [c for c in parts_df.columns if "diag" in c.lower() or "diagn" in c.lower() or "group" in c.lower()]
            print(f"[INFO] Loaded participants.tsv: {participants_path} -> shape={parts_df.shape}")
            if pid_cols:
                print(f"[INFO] participants has id column(s): {pid_cols[0]}")
            if diag_cols:
                print(f"[INFO] participants has diagnosis-like column(s): {', '.join(diag_cols)}")
        except Exception as e:
            print("[WARN] Could not read participants.tsv:", e)
            participants_present = False
    else:
        print("[INFO] No participants.tsv found at", participants_path)

    # Sample matching checks
    print("")
    print("=== Sample matching summary ===")
    print(f"Gene-expression samples (columns): {n_samples}")
    meta_sample_ids = []
    if meta_present:
        # get sample ids from metadata
        meta_sample_ids = gene_meta_df["sample_id"].astype(str).tolist()
        print(f"Gene metadata sample count: {len(meta_sample_ids)}")
    else:
        print("Gene metadata not present -> skipping sample alignment with metadata.")

    # compute intersections
    if meta_present:
        set_genes = set(gene_sample_ids)
        set_meta = set(meta_sample_ids)
        in_genes_not_meta = sorted(list(set_genes - set_meta))
        in_meta_not_genes = sorted(list(set_meta - set_genes))

        n_in_both = len(set_genes & set_meta)
        print(f"Samples present in both gene-matrix and metadata: {n_in_both}")
        if in_genes_not_meta:
            print(f"[WARN] Samples present in gene matrix but missing from metadata: {len(in_genes_not_meta)} (example 10): {in_genes_not_meta[:10]}")
        if in_meta_not_genes:
            print(f"[WARN] Samples present in metadata but missing from gene matrix: {len(in_meta_not_genes)} (example 10): {in_meta_not_genes[:10]}")

        # Write mismatched file if any
        if in_genes_not_meta or in_meta_not_genes:
            mismatches = pd.DataFrame({
                "in_genes_not_meta": pd.Series(in_genes_not_meta),
                "in_meta_not_genes": pd.Series(in_meta_not_genes)
            })
            out_mismatch = out_dir / "mismatched_samples.csv"
            mismatches.to_csv(out_mismatch, index=False)
            print(f"[INFO] Wrote mismatched samples to: {out_mismatch}")
            # treat as critical mismatch -> exit non-zero
            print("[ERROR] Sample mismatch between gene matrix and metadata. Fix metadata or re-run gene_feature_assembly.")
            sys.exit(3)
        else:
            print("[OK] Gene matrix and metadata sample sets match.")

    # If participants present, just report overlap with gene samples (not critical)
    if participants_present:
        pid_col = None
        for c in ["participant_id", "participant", "subject", "sub", "participant_id"]:
            if c in parts_df.columns:
                pid_col = c
                break
        if pid_col is None:
            # try any column that looks like id
            pid_col = parts_df.columns[0]
        participants_ids = parts_df[pid_col].astype(str).tolist()
        set_parts = set(participants_ids)
        set_genes = set(gene_sample_ids)
        overlap = set_parts & set_genes
        print(f"[INFO] participants.tsv samples: {len(participants_ids)}. Overlap with gene samples: {len(overlap)}")
        if len(overlap) == 0:
            print("[WARN] No overlap between brain participants and gene sample IDs (this may be expected for synthetic-only workflows).")
        else:
            # list small sample of non-overlap
            not_in_parts = sorted(list(set_genes - set_parts))
            not_in_genes = sorted(list(set_parts - set_genes))
            if not_in_parts:
                print(f"[WARN] gene samples not in participants.tsv: {len(not_in_parts)} (example 5): {not_in_parts[:5]}")
            if not_in_genes:
                print(f"[WARN] participant samples not in gene matrix: {len(not_in_genes)} (example 5): {not_in_genes[:5]}")
            # don't treat as critical

    print("")
    print("[DONE] Dataset sanity checks complete.")
    sys.exit(0)


if __name__ == "__main__":
    main()

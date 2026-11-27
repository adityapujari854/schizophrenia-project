#!/usr/bin/env python3
"""
scripts/map_snps_to_genes.py

Map SNPs from a GWAS file to genes using a GTF (GENCODE v19 recommended).

Outputs (default into data/processed/genetics/):
  - snp_gene_annotation.csv   : per-SNP rows with overlapping gene(s) (or None)
  - gene_snp_list.csv         : per-gene summary (n_snps, min_snp_p, snp_list)

Usage examples:
    python scripts/map_snps_to_genes.py \
        --gwas data/raw/genetics/daner_natgen_pgc_eas.xlsx \
        --gtf data/raw/genes/gencode.v19.chr_patch_hapl_scaff.annotation.gtf.gz \
        --out-dir data/processed/genetics --window 0

    # with 10kb flanking window around genes
    python scripts/map_snps_to_genes.py --gwas gw.csv --gtf gtf.gz --window 10000

Notes:
 - GWAS file may be Excel (.xlsx/.xls) or CSV/TSV. Script attempts to detect rsid/chr/pos/pval columns heuristically.
 - GTF should contain 'gene' feature entries (GENCODE). The parser extracts gene_id and gene_name attributes.
"""
from __future__ import annotations

import argparse
import gzip
import itertools
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# -----------------------
# Defaults
# -----------------------
PROJECT_ROOT = Path.cwd()
DEFAULT_GWAS = PROJECT_ROOT / "data" / "raw" / "genetics" / "daner_natgen_pgc_eas.xlsx"
DEFAULT_GTF = PROJECT_ROOT / "data" / "raw" / "genes" / "gencode.v19.chr_patch_hapl_scaff.annotation.gtf.gz"
DEFAULT_OUT = PROJECT_ROOT / "data" / "processed" / "genetics"

# -----------------------
# Utilities
# -----------------------
def read_gwas_table(path: Path) -> pd.DataFrame:
    """Read GWAS file (Excel or CSV/TSV) and return a DataFrame with minimal normalization."""
    if not path.exists():
        raise FileNotFoundError(f"GWAS file not found: {path}")
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        # infer separator using pandas engine
        df = pd.read_csv(path, sep=None, engine="python")
    if df.shape[0] == 0:
        raise ValueError("GWAS file appears empty.")
    return df


def guess_gwas_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    Heuristically find columns for rsid, chr, pos, pval in the GWAS DataFrame.
    Returns mapping: {'rsid': colname, 'chr': colname, 'pos': colname, 'pval': colname_or_none}
    """
    colmap: Dict[str, Optional[str]] = {"rsid": None, "chr": None, "pos": None, "pval": None}
    cols = list(df.columns)
    lower = [c.lower() for c in cols]

    for i, c in enumerate(cols):
        lc = lower[i]
        if colmap["rsid"] is None and (("rs" in lc and "id" in lc) or lc == "snp" or lc.startswith("rs")):
            colmap["rsid"] = c
        if colmap["chr"] is None and any(x in lc for x in ("chr", "chrom", "chromosome")):
            colmap["chr"] = c
        if colmap["pos"] is None and any(x in lc for x in ("pos", "bp", "position")):
            colmap["pos"] = c
        if colmap["pval"] is None and any(x in lc for x in ("pvalue", "p_value", "p.value", "p")):
            # choose only if it's probably a p-value column (not chromosome 'p' weirdness)
            # ensure not an integer-like 'p' that is chromosome index; still heuristics
            colmap["pval"] = c

    # Last-resort: if rsid not found but a column contains 'rs' tokens, select that column
    if colmap["rsid"] is None:
        for c in cols:
            sample_vals = df[c].astype(str).head(20).tolist()
            if any(re.match(r"^rs\d+$", s, flags=re.I) for s in sample_vals):
                colmap["rsid"] = c
                break

    if colmap["chr"] is None or colmap["pos"] is None:
        raise ValueError(f"Could not detect chr/pos columns automatically. Found columns: {cols}")

    return {k: v for k, v in colmap.items() if v is not None}


def normalize_chr(x: str) -> str:
    """Normalize chromosome string by stripping leading 'chr' and handling 'X','Y','MT' variants."""
    if x is None:
        return ""
    s = str(x).strip()
    s = re.sub(r"^chr", "", s, flags=re.I)
    s = s.replace("MT", "M")
    return s


def parse_gtf_genes(gtf_path: Path) -> pd.DataFrame:
    """
    Parse GTF and return DataFrame with columns:
      ['gene_id', 'gene_name', 'chrom', 'start', 'end', 'strand']
    Only rows with feature == 'gene' are returned.
    """
    if not gtf_path.exists():
        raise FileNotFoundError(f"GTF not found: {gtf_path}")
    rows = []
    opener = gzip.open if str(gtf_path).endswith(".gz") else open
    with opener(gtf_path, "rt", errors="ignore") as fh:
        for ln in fh:
            if ln.startswith("#") or not ln.strip():
                continue
            parts = ln.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            chrom, src, feature, start, end, score, strand, frame, attrs = parts
            if feature.lower() != "gene":
                continue
            # parse attributes e.g. key "value"; key2 "value2";
            attr_dict = {}
            for m in re.finditer(r'(\S+)\s+"([^"]+)"', attrs):
                attr_dict[m.group(1)] = m.group(2)
            gene_id = attr_dict.get("gene_id", "")
            gene_name = attr_dict.get("gene_name", gene_id)
            rows.append({
                "gene_id": gene_id,
                "gene_name": gene_name,
                "chrom": normalize_chr(chrom),
                "start": int(start),
                "end": int(end),
                "strand": strand
            })
    if not rows:
        raise ValueError("No gene entries parsed from GTF (check file and feature types).")
    gdf = pd.DataFrame(rows)
    return gdf


def map_snps_to_genes(gwas_df: pd.DataFrame, genes_df: pd.DataFrame, window: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Map SNPs to overlapping genes (with optional flanking `window` bp).
    Returns (snp_annotation_df, gene_snp_list_df).
    """
    # prepare genes grouped by chromosome for faster lookup
    genes_by_chr: Dict[str, List[dict]] = {}
    for chrom, group in genes_df.groupby("chrom"):
        # sort by start for potential small speedup
        genes_by_chr[chrom] = group.sort_values("start").to_dict("records")

    out_rows = []
    gene_to_snps = defaultdict(list)

    for idx, row in gwas_df.iterrows():
        rsid = str(row.get("rsid", "")) if "rsid" in row else ""
        chrom = normalize_chr(str(row["chr"]))
        pos = int(row["pos"])
        pval = row.get("pval", None) if "pval" in row else None

        matched = []
        for g in genes_by_chr.get(chrom, []):
            # overlap if pos in [start-window, end+window]
            if pos >= (g["start"] - window) and pos <= (g["end"] + window):
                matched.append(g)

        if matched:
            for g in matched:
                out_rows.append({
                    "rsid": rsid,
                    "chr": chrom,
                    "pos": pos,
                    "pval": pval,
                    "gene_id": g["gene_id"],
                    "gene_name": g["gene_name"],
                    "gene_start": g["start"],
                    "gene_end": g["end"],
                    "overlap_type": "within_window" if window > 0 else "within_gene"
                })
                gene_to_snps[g["gene_name"]].append({"rsid": rsid, "pos": pos, "pval": pval})
        else:
            out_rows.append({
                "rsid": rsid,
                "chr": chrom,
                "pos": pos,
                "pval": pval,
                "gene_id": None,
                "gene_name": None,
                "gene_start": None,
                "gene_end": None,
                "overlap_type": "none"
            })

    snp_annotation_df = pd.DataFrame(out_rows)

    # gene-level summaries
    gene_rows = []
    for gene_name, snps in gene_to_snps.items():
        pvals = [s["pval"] for s in snps if s["pval"] is not None]
        gene_rows.append({
            "gene": gene_name,
            "n_snps": len(snps),
            "min_snp_p": (min(pvals) if pvals else None),
            "snp_list": ";".join(sorted({s["rsid"] or f"{s['pos']}" for s in snps}))
        })
    gene_snp_df = pd.DataFrame(gene_rows).sort_values("n_snps", ascending=False)

    return snp_annotation_df, gene_snp_df


# -----------------------
# CLI entrypoint
# -----------------------
def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="map_snps_to_genes.py", description="Map GWAS SNPs to genes using GTF coordinates.")
    p.add_argument("--gwas", type=Path, default=DEFAULT_GWAS, help="GWAS file (Excel or CSV).")
    p.add_argument("--gtf", type=Path, default=DEFAULT_GTF, help="GENCODE GTF file (.gtf or .gtf.gz).")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help="Output directory for annotation CSVs.")
    p.add_argument("--window", type=int, default=0, help="Flanking window (bp) to extend gene boundaries for mapping.")
    p.add_argument("--dry-run", action="store_true", help="Do not write outputs; just print what would be done.")
    args = p.parse_args(argv)

    try:
        gwas_df = read_gwas_table(args.gwas)
    except Exception as e:
        print(f"[ERROR] Could not read GWAS file: {e}")
        return 2

    try:
        colmap = guess_gwas_columns(gwas_df)
    except Exception as e:
        print(f"[ERROR] Could not detect columns in GWAS file automatically: {e}")
        return 3

    # Build canonical dataframe with columns: rsid, chr, pos, pval
    try:
        canonical = pd.DataFrame()
        canonical["rsid"] = gwas_df[colmap.get("rsid")] if colmap.get("rsid") else gwas_df.index.astype(str)
        canonical["chr"] = gwas_df[colmap["chr"]].astype(str).apply(normalize_chr)
        canonical["pos"] = pd.to_numeric(gwas_df[colmap["pos"]], errors="coerce").astype("Int64")
        if colmap.get("pval"):
            canonical["pval"] = pd.to_numeric(gwas_df[colmap["pval"]], errors="coerce")
        else:
            canonical["pval"] = None
        canonical = canonical.dropna(subset=["pos"])
    except Exception as e:
        print(f"[ERROR] Failed to normalize GWAS columns: {e}")
        return 4

    if canonical.empty:
        print("[ERROR] No valid SNP rows after parsing GWAS file.")
        return 5

    try:
        genes_df = parse_gtf_genes(args.gtf)
    except Exception as e:
        print(f"[ERROR] Failed to parse GTF: {e}")
        return 6

    print(f"[INFO] GWAS SNPs parsed: {len(canonical)}")
    print(f"[INFO] Genes parsed from GTF: {len(genes_df)}")
    print(f"[INFO] Mapping with window = {args.window} bp")

    snp_ann_df, gene_snp_df = map_snps_to_genes(canonical, genes_df, window=args.window)

    if args.dry_run:
        print("[DRY RUN] Would write outputs to:", args.out_dir)
        print("[DRY RUN] SNP annotation sample:")
        print(snp_ann_df.head().to_string(index=False))
        print("[DRY RUN] Gene->SNP sample:")
        print(gene_snp_df.head().to_string(index=False))
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    snp_out = args.out_dir / "snp_gene_annotation.csv"
    gene_out = args.out_dir / "gene_snp_list.csv"

    snp_ann_df.to_csv(snp_out, index=False)
    gene_snp_df.to_csv(gene_out, index=False)

    print(f"[OK] Saved SNP->gene annotation -> {snp_out} (rows: {len(snp_ann_df)})")
    print(f"[OK] Saved gene-level SNP summary -> {gene_out} (rows: {len(gene_snp_df)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

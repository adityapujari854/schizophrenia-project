#!/usr/bin/env python3
"""
merge_gene_symbols_into_mapping.py

Attach real gene symbols (from a mapping CSV) to:
 - results/fusion_feature_component_mapping.csv  --> add gene_symbol column
 - results/fusion_top_permutation_importance.csv  --> add top_genes column (per PCA feature)

Usage (PowerShell):
  python .\scripts\merge_gene_symbols_into_mapping.py --mapping results\gene_symbol_mapping_new.csv

Assumptions:
 - mapping CSV columns: feature_index (int) , gene_symbol (str) [, method]
 - feature-component mapping path: results/fusion_feature_component_mapping.csv
 - top importance path: results/fusion_top_permutation_importance.csv
 - script writes augmented versions into results/ with "_augmented" suffix.
"""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mapping", type=str, default="results/gene_symbol_mapping_new.csv",
                   help="CSV mapping feature_index->gene_symbol")
    p.add_argument("--feature-map", type=str, default="results/fusion_feature_component_mapping.csv",
                   help="Component -> original feature mapping CSV")
    p.add_argument("--top-imp", type=str, default="results/fusion_top_permutation_importance.csv",
                   help="Top permutation importance CSV")
    p.add_argument("--out-dir", type=str, default="results",
                   help="Output directory for augmented CSVs")
    p.add_argument("--top-n", type=int, default=10,
                   help="How many original features per component to list into top_genes (for top-importance table)")
    return p.parse_args()

def load_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Expected file not found: {path}")
    return pd.read_csv(path)

def main():
    args = parse_args()
    mapping_p = Path(args.mapping)
    fmap_p = Path(args.feature_map)
    topimp_p = Path(args.top_imp)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load mapping of feature_index -> gene_symbol
    map_df = load_csv(mapping_p)
    if "feature_index" not in map_df.columns or "gene_symbol" not in map_df.columns:
        raise ValueError("Mapping CSV must contain columns: feature_index,gene_symbol")
    # normalize
    map_df = map_df.copy()
    map_df["feature_index"] = map_df["feature_index"].astype(int)
    map_dict = dict(zip(map_df["feature_index"].astype(int), map_df["gene_symbol"].astype(str)))

    # Load component mapping
    fmap_df = load_csv(fmap_p)
    fmap = fmap_df.copy()
    # ensure orig_mod / orig_index present
    if not {"feature_component","orig_mod","orig_index"}.issubset(set(fmap.columns)):
        raise ValueError(f"feature map CSV missing required columns. Found: {list(fmap.columns)}")

    # For rows where orig_mod == 'genes' (or 'gene'), attach gene_symbol via orig_index
    def lookup_symbol(row):
        if str(row["orig_mod"]).lower().startswith("gene"):
            idx = int(row["orig_index"])
            return map_dict.get(idx, pd.NA)
        return pd.NA

    fmap["gene_symbol"] = fmap.apply(lookup_symbol, axis=1)

    # Save augmented feature-component mapping
    out_map_p = out_dir / (fmap_p.stem + "_augmented.csv")
    fmap.to_csv(out_map_p, index=False)
    print(f"[OK] Wrote augmented component mapping to: {out_map_p} (rows={len(fmap)})")

    # Now attach gene symbols to top-importance features.
    # Load top importances
    top_df = load_csv(topimp_p)
    tdf = top_df.copy()

    # helper: from component name like "gPCA_0" or "bPCA_12" determine orig_mod ('genes' for gPCA)
    def component_to_mod_and_idx(comp_name):
        # expecting format like "gPCA_0", "bPCA_12", "sPCA_5"
        if "_" in comp_name:
            prefix, num = comp_name.split("_", 1)
            prefix = prefix.lower()
            try:
                cidx = int(num)
            except:
                cidx = None
            if prefix.startswith("gpca") or prefix.startswith("g"):
                return "genes", cidx
            if prefix.startswith("bpca") or prefix.startswith("b"):
                return "brain", cidx
            if prefix.startswith("spca") or prefix.startswith("s"):
                return "snps", cidx
        return None, None

    # Build a helper dict: for each component (feature_component) get top original gene symbols (from fmap) ordered by rank_within_component
    # expected fmap columns: feature_component, orig_mod, orig_index, loading_abs, rank (or rank_within_component)
    rank_col = None
    for cand in ("rank_within_component","rank","rank_within_comp","rank_in_component"):
        if cand in fmap.columns:
            rank_col = cand
            break
    if rank_col is None:
        # try to use loading_abs ordering
        fmap_sorted = fmap.copy()
    else:
        fmap_sorted = fmap.sort_values([ "feature_component", rank_col ])

    comp2orig = {}
    for comp, group in fmap_sorted.groupby("feature_component"):
        # select only gene rows and map to gene symbols
        gene_rows = group[group["orig_mod"].str.lower().str.startswith("gene")]
        # order by rank_col if exists else by loading_abs desc
        if rank_col:
            gene_rows = gene_rows.sort_values(rank_col)
        else:
            if "loading_abs" in gene_rows.columns:
                gene_rows = gene_rows.sort_values("loading_abs", ascending=False)
        # map orig_index -> symbol
        syms = []
        for _, r in gene_rows.iterrows():
            idx = int(r["orig_index"])
            sym = map_dict.get(idx)
            if sym:
                syms.append(str(sym))
        comp2orig[comp] = syms

    # For each top feature row, add a 'top_genes' column composed of top N gene symbols from the corresponding component
    top_n = int(args.top_n)
    def gather_top_genes(feat_name):
        # handle possible suffixes/whitespace
        name = str(feat_name).strip()
        # direct key match in comp2orig
        if name in comp2orig:
            return ";".join(comp2orig[name][:top_n]) if comp2orig[name] else ""
        # sometimes features listed as e.g. "gPCA_0" or "gPCA_0 " etc; normalize
        k = name
        if k in comp2orig:
            return ";".join(comp2orig[k][:top_n]) if comp2orig[k] else ""
        return ""

    tdf["top_genes"] = tdf["feature"].apply(gather_top_genes)

    out_top_p = out_dir / (topimp_p.stem + "_augmented.csv")
    tdf.to_csv(out_top_p, index=False)
    print(f"[OK] Wrote augmented top-importance file to: {out_top_p} (rows={len(tdf)})")

    # Also produce a small summary file mapping gene_symbol -> list of components where it appears (for quick lookup)
    comp_list = []
    for comp, syms in comp2orig.items():
        for i, sym in enumerate(syms):
            comp_list.append({"feature_component": comp, "gene_symbol": sym, "rank": i+1})
    comp_map_df = pd.DataFrame(comp_list)
    if not comp_map_df.empty:
        out_comp_map = out_dir / "gene_to_components.csv"
        comp_map_df.to_csv(out_comp_map, index=False)
        print(f"[OK] Wrote gene->component summary to: {out_comp_map} (rows={len(comp_map_df)})")
    else:
        print("[WARN] No gene->component mappings produced (no gene rows found in feature mapping).")

    print("[DONE] merge_gene_symbols_into_mapping completed.")

if __name__ == "__main__":
    main()
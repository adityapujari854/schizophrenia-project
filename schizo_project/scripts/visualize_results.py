#!/usr/bin/env python3
"""
visualize_results.py

Generate summary plots from experiment outputs.
""" 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "viz"
PLOTS_DIR.mkdir(exist_ok=True)

def plot_auc_vs_effect(summary_csv):
    df = pd.read_csv(summary_csv)
    # expand JSON metrics
    metrics = df["fusion_metrics_summary"].apply(json.loads).apply(pd.Series)
    df = pd.concat([df, metrics], axis=1)

    plt.figure(figsize=(8,6))
    for model in ["LogisticRegression_test", "RandomForest_test", "XGBoost_test"]:
        sns.lineplot(data=df, x="effect", y=model, marker="o", label=model)
    plt.xlabel("Injected effect size")
    plt.ylabel("Test AUC")
    plt.title("Effect size vs Model Performance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "effect_vs_auc.png")
    plt.close()


def plot_recovery_bar(recovery_csv):
    """
    Robust plotting for recovery results.

    - If 'recovered' column present, use it.
    - Otherwise: compute 'recovered' by checking orig_index against
      injected indices found in any signal_summary.json files under the repo.
    """
    recovery_csv = Path(recovery_csv)
    if not recovery_csv.exists():
        raise FileNotFoundError(f"Recovery CSV not found: {recovery_csv}")

    df = pd.read_csv(recovery_csv)

    # column checks
    if "recovered" not in df.columns:
        # need orig_mod and orig_index to compute
        if "orig_index" not in df.columns:
            raise KeyError("'orig_index' column missing from recovery CSV; cannot compute recovery.")
        print("[WARN] 'recovered' column missing — attempting to compute from local signal_summary.json files...")

        # find all signal_summary.json files in repository (data/* directories)
        repo_root = Path.cwd()
        summary_files = list(repo_root.glob("**/signal_summary.json"))
        injected_genes = set()
        injected_brain = set()
        injected_snps = set()

        for sf in summary_files:
            try:
                s = json.loads(sf.read_text())
            except Exception:
                continue
            # keys may be 'gene_signal_examples' or 'n_gene_signal' etc.
            # prefer full lists if present, else skip
            for k in ("gene_signal_examples", "gene_signal_indices", "genes_injected"):
                if k in s and isinstance(s[k], (list, tuple)):
                    injected_genes.update(int(x) for x in s[k])
            for k in ("brain_signal_examples", "brain_signal_indices", "brain_injected"):
                if k in s and isinstance(s[k], (list, tuple)):
                    injected_brain.update(int(x) for x in s[k])
            for k in ("snp_signal_examples", "snp_signal_indices", "snps_injected"):
                if k in s and isinstance(s[k], (list, tuple)):
                    injected_snps.update(int(x) for x in s[k])

        # If no explicit lists found, try numeric 'n_*' fields (can't recover indices)
        if not (injected_genes or injected_brain or injected_snps):
            print("[WARN] No signal_summary.json with explicit index lists found. 'recovered' cannot be computed reliably.")
            # fallback: add 'recovered' column filled with False
            df["recovered"] = False
        else:
            # compute recovered boolean per row
            def _is_recovered(row):
                mod = str(row.get("orig_mod", "")).strip().lower()
                try:
                    idx = int(row["orig_index"])
                except Exception:
                    return False
                if mod.startswith("gene") or mod == "genes":
                    return idx in injected_genes
                if mod.startswith("brain"):
                    return idx in injected_brain
                if mod.startswith("snp") or mod.startswith("snps"):
                    return idx in injected_snps
                # unknown modality: check all
                return (idx in injected_genes) or (idx in injected_brain) or (idx in injected_snps)

            df["recovered"] = df.apply(_is_recovered, axis=1)
            print(f"[INFO] Computed 'recovered' flags using {len(summary_files)} signal_summary.json files.")
            print(f"  injected_genes={len(injected_genes)}, injected_brain={len(injected_brain)}, injected_snps={len(injected_snps)}")

    # now plotting
    # if 'experiment' exists, group by experiment; otherwise group by modality
    if "experiment" in df.columns:
        agg = df.groupby("experiment")["recovered"].mean().reset_index()
        plt.figure(figsize=(8, 5))
        sns.barplot(data=agg, x="experiment", y="recovered")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Recovery rate (fraction)")
        plt.title("Recovery across Experiments")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "recovery_by_experiment.png")
        plt.close()
    else:
        # group by orig_mod (orig_mod should exist)
        if "orig_mod" not in df.columns:
            raise KeyError("'orig_mod' column missing from recovery CSV; cannot group by modality.")
        agg = df.groupby("orig_mod")["recovered"].mean().reset_index()
        plt.figure(figsize=(6, 5))
        sns.barplot(data=agg, x="orig_mod", y="recovered")
        plt.ylabel("Recovery rate (fraction)")
        plt.title("Recovery by Modality")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "recovery_by_modality.png")
        plt.close()

    # Save possibly-updated CSV (so next time script sees recovered column)
    try:
        out_csv = Path("results") / recovery_csv.name
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"[INFO] Wrote augmented recovery CSV with 'recovered' column to: {out_csv}")
    except Exception as e:
        print(f"[WARN] Could not write augmented CSV: {e}")


def plot_top_features(importance_csv):
    df = pd.read_csv(importance_csv).sort_values("importance_mean", ascending=False).head(20)
    plt.figure(figsize=(8,6))
    sns.barplot(data=df, x="importance_mean", y="feature", orient="h")
    plt.xlabel("Permutation Importance")
    plt.ylabel("Top Features")
    plt.title("Top 20 Fusion Features")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "top_features.png")
    plt.close()

def main():
    print("[INFO] Generating visualization plots...")
    plot_auc_vs_effect(RESULTS_DIR / "experiment_summary.csv")
    plot_top_features(RESULTS_DIR / "fusion_top_permutation_importance.csv")
    # recovery is optional per-experiment
    recovery_file = RESULTS_DIR / "recovery_matches.csv"
    if recovery_file.exists():
        plot_recovery_bar(recovery_file)
    print("[OK] Plots saved to", PLOTS_DIR)

if __name__ == "__main__":
    main()

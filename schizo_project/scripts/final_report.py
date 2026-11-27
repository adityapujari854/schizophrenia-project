#!/usr/bin/env python3
"""
final_report.py (improved)

Build a consolidated final report from experiment outputs.

Saves:
 - results/final_report/summary_experiments.csv
 - results/final_report/metrics_agg.csv
 - results/final_report/recovery_summary.csv
 - results/final_report/top_features.csv (if exists)
 - results/final_report/*.png  (plots)
 - results/final_report/report.md  (simple human-readable report)

Usage (PowerShell):
    python .\scripts\final_report.py
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import textwrap
import sys
import json
from datetime import datetime

# Config / paths
PROJECT_ROOT = Path.cwd()
RESULTS_DIR = PROJECT_ROOT / "results"
OUT_DIR = RESULTS_DIR / "final_report"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "experiment_summary": RESULTS_DIR / "experiment_summary.csv",
    "recovery_matches": RESULTS_DIR / "recovery_matches.csv",
    "fusion_metrics": RESULTS_DIR / "fusion_metrics.csv",
    "gene_metrics": RESULTS_DIR / "gene_metrics.csv",
    "brain_metrics": RESULTS_DIR / "brain_metrics.csv",
    "fusion_top_importance": RESULTS_DIR / "fusion_top_permutation_importance.csv",
    "feature_map": RESULTS_DIR / "fusion_feature_component_mapping.csv",
}

PNG_OPTS = dict(dpi=150, bbox_inches="tight")


# ---- helpers ----
def load_if_exists(path: Path, **kwargs):
    if path.exists():
        try:
            return pd.read_csv(path, **kwargs)
        except Exception as e:
            print(f"[WARN] Failed to read {path}: {e}")
            return None
    else:
        print(f"[INFO] Not found: {path.name} — skipping.")
        return None


def safe_save_df(df: pd.DataFrame, name: str):
    dest = OUT_DIR / name
    df.to_csv(dest, index=False)
    print(f"[OK] Wrote {dest}")


# ---- recovery helper: infer 'recovered' if missing ----
def infer_recovered_flags_from_summaries(recovery_df: pd.DataFrame):
    """
    If recovery_df lacks a 'recovered' column, attempt to infer it using:
     - 'match_found' column if available
     - otherwise, read signal_summary.json from each experiment out_dir to get injected indices
       and mark rows as recovered if orig_index in injected set and orig_mod matches.
    This is best-effort and will leave rows NA if cannot infer.
    """
    df = recovery_df.copy()
    if "recovered" in df.columns:
        return df

    # use match_found if present
    if "match_found" in df.columns:
        df["recovered"] = df["match_found"].astype(int)
        print("[INFO] Derived 'recovered' from 'match_found' column.")
        return df

    # If out_dir and orig_index exist, try to read signal_summary.json for each out_dir
    if "out_dir" in df.columns and "orig_index" in df.columns and "orig_mod" in df.columns:
        recovered = []
        grouped = df.groupby("out_dir")
        for out_dir, g in grouped:
            out_path = Path(str(out_dir))
            if not out_path.exists():
                alt = (PROJECT_ROOT / out_dir)
                if alt.exists():
                    out_path = alt
            # load injected examples
            sfile = out_path / "signal_summary.json"
            injected_by_mod = {"genes": set(), "brain": set(), "snps": set()}
            if sfile.exists():
                try:
                    js = json.loads(sfile.read_text(encoding="utf-8"))
                    # keys may vary (use fallbacks)
                    injected_by_mod["genes"] = set(js.get("gene_signal_examples", js.get("gene_signal_indices", [])))
                    injected_by_mod["brain"] = set(js.get("brain_signal_examples", js.get("brain_signal_indices", [])))
                    injected_by_mod["snps"] = set(js.get("snp_signal_examples", js.get("snp_signal_indices", [])))
                except Exception:
                    injected_by_mod = {"genes": set(), "brain": set(), "snps": set()}
            else:
                injected_by_mod = {"genes": set(), "brain": set(), "snps": set()}

            for _, row in g.iterrows():
                orig_mod = str(row["orig_mod"]).strip().lower()
                try:
                    orig_idx = int(row["orig_index"])
                except Exception:
                    recovered.append(0)
                    continue
                if orig_mod.startswith("gene"):
                    recovered.append(1 if orig_idx in injected_by_mod.get("genes", set()) else 0)
                elif orig_mod.startswith("brain"):
                    recovered.append(1 if orig_idx in injected_by_mod.get("brain", set()) else 0)
                elif orig_mod.startswith("snp"):
                    recovered.append(1 if orig_idx in injected_by_mod.get("snps", set()) else 0)
                else:
                    recovered.append(0)
        df["recovered"] = recovered
        print("[INFO] Computed 'recovered' flags using available signal_summary.json files (best-effort).")
        return df

    # fallback: create recovered column filled with 0 and warn
    df["recovered"] = 0
    print("[WARN] Could not infer 'recovered' from recovery_matches; filled with zeros.")
    return df


# ---- aggregation functions ----
def aggregate_experiment_summary(df):
    """Clean and return experiment_summary with parsed fusion_metrics_summary JSON expanded."""
    if df is None:
        return None
    df = df.copy()
    if "fusion_metrics_summary" in df.columns:
        def parse_json_cell(x):
            try:
                return json.loads(x) if pd.notna(x) and x != "" else {}
            except Exception:
                try:
                    return eval(x) if pd.notna(x) and x != "" else {}
                except Exception:
                    return {}
        parsed = df["fusion_metrics_summary"].apply(parse_json_cell)
        all_keys = set(k for d in parsed for k in d.keys())
        for k in sorted(all_keys):
            df[f"fusion_{k}"] = parsed.apply(lambda d: d.get(k, np.nan))
    if "timestamp" in df.columns:
        df["ts_readable"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
    return df


def summarize_metrics(gene_df, brain_df, fusion_df):
    """Return a unified metrics table with model-level rows and basic stats."""
    rows = []
    if gene_df is not None:
        try:
            for _, r in gene_df.iterrows():
                rows.append({
                    "modality": "genes",
                    "model": str(r.get("model", "")),
                    "roc_auc": float(r.get("roc_auc", np.nan))
                })
        except Exception:
            pass
    if brain_df is not None:
        try:
            for _, r in brain_df.iterrows():
                rows.append({
                    "modality": "brain",
                    "model": str(r.get("model", "")),
                    "roc_auc": float(r.get("test_auc", r.get("roc_auc", np.nan)))
                })
        except Exception:
            pass
    if fusion_df is not None:
        try:
            for _, r in fusion_df.iterrows():
                model = str(r.get("model", "fusion"))
                auc = r.get("test_auc", r.get("cv_auc_mean", np.nan))
                rows.append({
                    "modality": "fusion",
                    "model": model,
                    "roc_auc": float(auc) if pd.notna(auc) else np.nan
                })
        except Exception:
            pass
    if not rows:
        return None
    return pd.DataFrame(rows)


def recovery_summary(recovery_df):
    """Aggregate recovery statistics by experiment/out_dir or orig_mod."""
    if recovery_df is None:
        return None
    df = recovery_df.copy()
    if "recovered" not in df.columns and "match_found" in df.columns:
        df["recovered"] = df["match_found"].astype(int)
    group_cols = []
    if "out_dir" in df.columns:
        group_cols = ["out_dir"]
    elif "experiment" in df.columns:
        group_cols = ["experiment"]
    else:
        group_cols = []
    if group_cols:
        agg = df.groupby(group_cols)["recovered"].mean().reset_index().rename(columns={"recovered": "frac_recovered"})
        return agg
    if "orig_mod" in df.columns:
        agg = df.groupby("orig_mod")["recovered"].mean().reset_index().rename(columns={"recovered": "frac_recovered"})
        return agg
    return None


# ---- plotting ----
def plot_auc_boxplot(metrics_df: pd.DataFrame):
    if metrics_df is None or metrics_df.empty:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    df = metrics_df.dropna(subset=["roc_auc"])
    if df.empty:
        return None
    modalities = df["modality"].unique()
    data = [df[df["modality"] == m]["roc_auc"].values for m in modalities]
    ax.boxplot(data, labels=modalities)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("AUC distribution by modality")
    ax.set_ylabel("ROC AUC")
    out = OUT_DIR / "auc_by_modality.png"
    fig.savefig(out, **PNG_OPTS)
    plt.close(fig)
    print(f"[OK] Saved {out}")
    return out


def plot_recovery_bar(recovery_df):
    if recovery_df is None or recovery_df.empty:
        return None
    if "frac_recovered" in recovery_df.columns and ("out_dir" in recovery_df.columns or "experiment" in recovery_df.columns):
        df = recovery_df.copy()
        label_col = "out_dir" if "out_dir" in df.columns else "experiment"
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(df[label_col].astype(str), df["frac_recovered"])
        ax.set_xticklabels(df[label_col].astype(str), rotation=45, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Fraction recovered (top-K)")
        ax.set_title("Recovery fraction by experiment")
        out = OUT_DIR / "recovery_by_experiment.png"
        fig.savefig(out, **PNG_OPTS)
        plt.close(fig)
        print(f"[OK] Saved {out}")
        return out
    if "orig_mod" in recovery_df.columns and "recovered" in recovery_df.columns:
        df = recovery_df.groupby("orig_mod")["recovered"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(df["orig_mod"], df["recovered"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Fraction recovered")
        ax.set_title("Recovery by modality (orig_mod)")
        out = OUT_DIR / "recovery_by_modality.png"
        fig.savefig(out, **PNG_OPTS)
        plt.close(fig)
        print(f"[OK] Saved {out}")
        return out
    return None


def plot_top_importances(imp_df, top_n=30):
    if imp_df is None or imp_df.empty:
        return None
    df = imp_df.copy().sort_values("importance_mean", ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(6, min(6, 0.25 * len(df) + 1)))
    ax.barh(df["feature"].astype(str)[::-1], df["importance_mean"][::-1])
    ax.set_xlabel("Permutation importance (mean)")
    ax.set_title(f"Top {top_n} fusion features (permutation importance)")
    out = OUT_DIR / "fusion_top_importances.png"
    fig.savefig(out, **PNG_OPTS)
    plt.close(fig)
    print(f"[OK] Saved {out}")
    return out


def write_markdown_report(summary_df, metrics_agg, recovery_agg, top_features_path, plots):
    md = []
    md.append("# Final experiment report")
    md.append(f"_Generated: {datetime.now().isoformat()}_")
    md.append("")
    md.append("## Summary of experiments")
    if summary_df is not None:
        md.append(f"- Experiments table saved to `results/final_report/summary_experiments.csv` ({len(summary_df)} rows).")
    else:
        md.append("- No `experiment_summary.csv` found; skipping experiment table.")
    md.append("")
    md.append("## Aggregated metrics")
    if metrics_agg is not None:
        md.append(f"- Aggregated metrics saved to `results/final_report/metrics_agg.csv`")
        md.append("")
        md.append("### AUC by modality (boxplot)")
        if plots.get("auc"):
            md.append(f"![AUC by modality](./{plots['auc'].name})")
    else:
        md.append("- No metrics found.")
    md.append("")
    md.append("## Recovery")
    if recovery_agg is not None:
        md.append("- Recovery summary saved to `results/final_report/recovery_summary.csv`")
        if plots.get("recovery"):
            md.append(f"![Recovery]({plots['recovery'].name})")
    else:
        md.append("- No recovery data found.")
    md.append("")
    md.append("## Top features")
    if top_features_path:
        md.append(f"- Top permutation importances saved to `results/final_report/top_features.csv`")
        if plots.get("top_imp"):
            md.append(f"![Top importances]({plots['top_imp'].name})")
    else:
        md.append("- No feature importance data found.")
    md.append("")
    md.append("## Notes / Next steps")
    md.append(textwrap.dedent("""\
        - If you used synthetic signal datasets, consider running the `experiment_runner.py` with a broader grid
          to sweep effect sizes / number of signal features and then re-run this report to compare recoverability.
        - If SHAP failed previously for your model, permutation importances were used — they are slower but robust.
        - You can extend this report by exporting the top recovered original features (from the mapping CSV).
    """))
    md_path = OUT_DIR / "report.md"
    md_text = "\n\n".join(md)
    md_path.write_text(md_text, encoding="utf-8")
    print(f"[OK] Wrote {md_path}")


# ---- main driver ----
def main():
    print("[INFO] Loading available result files...")
    exp_df = load_if_exists(FILES["experiment_summary"])
    rec_df = load_if_exists(FILES["recovery_matches"])
    fusion_df = load_if_exists(FILES["fusion_metrics"])
    gene_df = load_if_exists(FILES["gene_metrics"])
    brain_df = load_if_exists(FILES["brain_metrics"])
    imp_df = load_if_exists(FILES["fusion_top_importance"])
    fmap_df = load_if_exists(FILES["feature_map"])

    # Aggregate experiment summary
    if exp_df is not None:
        exp_clean = aggregate_experiment_summary(exp_df)
        safe_save_df(exp_clean, "summary_experiments.csv")
    else:
        exp_clean = None

    # Recovery: ensure 'recovered' present (try automatic inference)
    if rec_df is not None:
        if "recovered" not in rec_df.columns:
            rec_df = infer_recovered_flags_from_summaries(rec_df)
        safe_save_df(rec_df, "recovery_matches_augmented.csv")
    rec_agg = recovery_summary(rec_df)

    # Aggregate metrics across modalities
    metrics_agg = summarize_metrics(gene_df, brain_df, fusion_df)
    if metrics_agg is not None:
        safe_save_df(metrics_agg, "metrics_agg.csv")

    # Recovery summary
    if rec_agg is not None:
        safe_save_df(rec_agg, "recovery_summary.csv")

    # Top feature importances
    top_features_path = None
    if imp_df is not None:
        imp_df_clean = imp_df.rename(columns=lambda c: c.strip())
        safe_save_df(imp_df_clean, "top_features.csv")
        top_features_path = OUT_DIR / "top_features.csv"

    # Plots
    plots = {}
    try:
        plots["auc"] = plot_auc_boxplot(metrics_agg)
    except Exception as e:
        print(f"[WARN] Could not make AUC plot: {e}")
    try:
        plots["recovery"] = plot_recovery_bar(rec_agg if rec_agg is not None else rec_df)
    except Exception as e:
        print(f"[WARN] Could not make recovery plot: {e}")
    try:
        plots["top_imp"] = plot_top_importances(imp_df, top_n=40)
    except Exception as e:
        print(f"[WARN] Could not make top importances plot: {e}")

    # Save a consolidated small JSON summary
    summary_record = {
        "generated_at": datetime.now().isoformat(),
        "experiment_count": int(len(exp_df)) if exp_df is not None else 0,
        "metrics_rows": int(len(metrics_agg)) if metrics_agg is not None else 0,
        "recovery_rows": int(len(rec_df)) if rec_df is not None else 0,
        "top_features": int(len(imp_df)) if imp_df is not None else 0,
    }
    with open(OUT_DIR / "report_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary_record, fh, indent=2)
    print(f"[OK] Wrote report summary JSON")

    # Write human-readable markdown report
    write_markdown_report(exp_clean, metrics_agg, rec_agg, top_features_path, plots)

    print("[DONE] Final report written to:", OUT_DIR)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print("ERROR in final_report:", e)
        sys.exit(1)

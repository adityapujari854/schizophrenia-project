#!/usr/bin/env python3
"""
aggregate_experiments.py

Reads results/experiment_summary_clean.csv (or experiment_summary.csv) and:
 - extracts fusion_metrics_summary JSON (if present) and converts to columns
 - creates a summary CSV results/experiment_agg.csv
 - plots effect vs fusion RandomForest_test (if available)

Requires: pandas, matplotlib
"""
from pathlib import Path
import pandas as pd
import ast
import json
import matplotlib.pyplot as plt

ROOT = Path.cwd()
RESULTS = ROOT / "results"
SRC = RESULTS / "experiment_summary_clean.csv"
if not SRC.exists():
    SRC = RESULTS / "experiment_summary.csv"
    if not SRC.exists():
        raise SystemExit(f"No experiment_summary found at {RESULTS}")

df = pd.read_csv(SRC)
print(f"[INFO] Loaded {len(df)} experiment records from {SRC}")

# parse fusion_metrics_summary column if present
def parse_metrics(s):
    try:
        if pd.isna(s) or s == "":
            return {}
        # some entries are JSON-like strings; try json.loads, then ast.literal_eval
        try:
            return json.loads(s)
        except Exception:
            return ast.literal_eval(s)
    except Exception:
        return {}

metrics_rows = []
for _, r in df.iterrows():
    metrics = parse_metrics(r.get("fusion_metrics_summary","") )
    # prefer Fusion test metrics names if available
    row = {
        "out_dir": r.get("out_dir",""),
        "n_gene_signal": int(r.get("n_gene_signal") or 0),
        "n_brain_signal": int(r.get("n_brain_signal") or 0),
        "effect": float(r.get("effect") or 0.0),
        "fraction": float(r.get("fraction") or 0.0),
        "seed": int(r.get("seed") or 0),
    }
    # add some metric fields if present
    # metrics may contain "RandomForest_test" or nested dicts; handle gracefully
    for k in ("LogisticRegression_test","RandomForest_test","XGBoost_test"):
        row[k] = metrics.get(k, None)
    # also allow older shape where metrics stored as dict-of-key->score
    if not any(row[k] for k in ("LogisticRegression_test","RandomForest_test","XGBoost_test")):
        # maybe metrics is a dict of test names
        for mk,v in metrics.items():
            # if mk contains 'RandomForest' use it
            if "RandomForest" in mk:
                row["RandomForest_test"] = v
            if "LogisticRegression" in mk:
                row["LogisticRegression_test"] = v
            if "XGBoost" in mk or "xgb" in mk:
                row["XGBoost_test"] = v
    metrics_rows.append(row)

agg = pd.DataFrame(metrics_rows)
out_csv = RESULTS / "experiment_agg.csv"
agg.to_csv(out_csv, index=False)
print(f"[OK] Wrote aggregated file: {out_csv}")

# Plot effect vs RandomForest_test if available
if agg["RandomForest_test"].notna().any():
    plt.figure(figsize=(6,4))
    plt.scatter(agg["effect"], agg["RandomForest_test"], c='tab:blue')
    plt.xlabel("Injected effect size")
    plt.ylabel("Fusion RandomForest test AUC")
    plt.title("Effect size vs Fusion RF AUC")
    plt.grid(True)
    plt.tight_layout()
    p = RESULTS / "effect_vs_fusion_rf_auc.png"
    plt.savefig(p)
    print(f"[OK] Saved plot: {p}")
else:
    print("[INFO] No RandomForest_test metrics found to plot.")

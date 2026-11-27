#!/usr/bin/env python3
"""
experiment_runner.py

Orchestrate signal injection -> fusion training -> explanation -> recovery
for many parameter combinations, with resume capability.

Usage examples (PowerShell):

# Single config run (recommended):
python .\scripts\experiment_runner.py --n-gene-signal 100 --n-brain-signal 40 --effect 1.4 --fraction 0.8 --out-dir data\synthetic_signal_exp1 --copy-all --overwrite

# Run a grid from CSV (columns: out_dir,n_gene_signal,n_brain_signal,n_snp_signal,effect,fraction,seed):
python .\scripts\experiment_runner.py --grid-csv configs\exp_grid.csv

# Quick dev (skips heavy explain/recovery by using --skip-explain/--skip-recovery):
python .\scripts\experiment_runner.py --n-gene-signal 50 --n-brain-signal 20 --effect 1.0 --fraction 0.8 --out-dir data\synthetic_signal_quick --copy-all --overwrite --quick

Notes:
- This script calls the repository's other scripts using the same Python interpreter running this script (sys.executable).
- Resume logic:
    * injection done if <out_dir>/signal_summary.json exists
    * fusion training done if results/fusion_metrics.csv exists AND models/fusion_rf.pkl exists
    * explanation done if results/fusion_top_permutation_importance.csv exists
    * recovery done if results/recovery_matches.csv exists
- It writes results/experiment_summary.csv with one row per experiment.

"""
from pathlib import Path
import argparse
import subprocess
import sys
import csv
import json
import time
import shutil
import pandas as pd

# ----------------- Helpers -----------------
ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
PLOTS_DIR = ROOT / "plots"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(MODELS_DIR).mkdir(parents=True, exist_ok=True)
(PLOTS_DIR).mkdir(parents=True, exist_ok=True)

PY = sys.executable  # use same python interpreter

def run_cmd(cmd_list, cwd=ROOT, log_path=None, env=None):
    """Run subprocess command, stream stdout/stderr to log file and console.
       Returns (returncode, duration_seconds). Raises on abnormal returncode only if fail_ok is False by caller."""
    start = time.time()
    print(f"[RUN] {' '.join(cmd_list)}")
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logfh = open(log_path, "a", encoding="utf-8")
    else:
        logfh = None
    try:
        proc = subprocess.Popen(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=str(cwd), env=env, text=True)
        # stream
        for line in proc.stdout:
            s = line.rstrip("\n")
            print(s)
            if logfh:
                logfh.write(s + "\n")
        proc.wait()
        rc = proc.returncode
    finally:
        if logfh:
            logfh.close()
    duration = time.time() - start
    print(f"[DONE] rc={rc} time={duration:.1f}s")
    return rc, duration

def ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def file_exists(path: Path):
    return path.exists()

def read_metrics_from_results():
    """Read fusion_metrics.csv if exists and return dict of metrics (first row)."""
    fm = RESULTS_DIR / "fusion_metrics.csv"
    if not fm.exists():
        return {}
    try:
        df = pd.read_csv(fm)
        # prefer test row lines named 'logreg', 'rf', 'xgb' or model columns if present
        # We'll return a simple mapping model->test_auc if available.
        out = {}
        if "test_auc" in df.columns and "model" in df.columns:
            for _, r in df.iterrows():
                if pd.notna(r.get("test_auc")):
                    out[str(r["model"])] = float(r["test_auc"])
        else:
            # fallback: return full CSV as JSON string
            out["fusion_metrics_csv"] = df.to_dict(orient="records")
        return out
    except Exception as e:
        return {"error": f"could not read fusion metrics: {e}"}

# ----------------- Steps checks -----------------
def injection_done(out_dir: Path):
    return (out_dir / "signal_summary.json").exists()

def fusion_training_done():
    # training considered done if at least fusion_rf.pkl exists AND fusion_metrics.csv exists
    return (MODELS_DIR / "fusion_rf.pkl").exists() and (RESULTS_DIR / "fusion_metrics.csv").exists()

def explanation_done():
    return (RESULTS_DIR / "fusion_top_permutation_importance.csv").exists()

def recovery_done():
    return (RESULTS_DIR / "recovery_matches.csv").exists()

# ----------------- Runner logic -----------------
def run_injection(cfg, base_log):
    out_dir = Path(cfg["out_dir"])
    args = [
        PY, str(SCRIPTS_DIR / "inject_signal.py"),
        "--data-dir", str(cfg.get("base_data_dir", "data/synthetic")),
        "--out-dir", str(out_dir),
        "--n-gene-signal", str(cfg.get("n_gene_signal", 50)),
        "--n-brain-signal", str(cfg.get("n_brain_signal", 20)),
        "--n-snp-signal", str(cfg.get("n_snp_signal", 0)),
        "--effect", str(cfg.get("effect", 0.8)),
        "--fraction", str(cfg.get("fraction", 0.8)),
        "--seed", str(cfg.get("seed", 42))
    ]
    if cfg.get("copy_all", True):
        args.append("--copy-all")
    if cfg.get("overwrite", False):
        args.append("--overwrite")
    # run
    logp = base_log.parent / f"{out_dir.name}_inject.log"
    rc, dur = run_cmd(args, log_path=logp)
    return rc == 0

def run_fusion(cfg, base_log, quick=False):
    out_dir = Path(cfg["out_dir"])
    args = [PY, str(SCRIPTS_DIR / "train_fusion_model.py"), "--data-dir", str(out_dir)]
    if quick or cfg.get("quick", False):
        args.append("--quick")
    # allow user override of PCA dims or n-estimators via cfg
    if "n_gene_pca" in cfg:
        args += ["--n-gene-pca", str(cfg["n_gene_pca"])]
    if "n_brain_pca" in cfg:
        args += ["--n-brain-pca", str(cfg["n_brain_pca"])]
    if "n_snp_pca" in cfg:
        args += ["--n-snp-pca", str(cfg["n_snp_pca"])]
    if cfg.get("no_xgb", False):
        args += ["--no-xgb"]
    logp = base_log.parent / f"{Path(cfg['out_dir']).name}_fusion.log"
    rc, dur = run_cmd(args, log_path=logp)
    return rc == 0

def run_explain(cfg, base_log, top=200, force_interventional=False):
    out_dir = Path(cfg["out_dir"])
    model_path = MODELS_DIR / "fusion_rf.pkl"
    if not model_path.exists():
        # fallback: use any fusion model present
        candidates = list(MODELS_DIR.glob("fusion*.pkl"))
        if candidates:
            model_path = candidates[0]
    args = [PY, str(SCRIPTS_DIR / "explain_fusion_model.py"), "--model", str(model_path), "--data-dir", str(out_dir), "--top", str(top)]
    if force_interventional:
        args.append("--force-interventional")
    logp = base_log.parent / f"{out_dir.name}_explain.log"
    rc, dur = run_cmd(args, log_path=logp)
    return rc == 0

def run_recovery(cfg, base_log, top_k=200, top_n=30):
    out_dir = Path(cfg["out_dir"])
    args = [PY, str(SCRIPTS_DIR / "check_recovery.py"), "--data-dir", str(out_dir), "--top-k", str(top_k), "--top-n", str(top_n)]
    logp = base_log.parent / f"{out_dir.name}_recovery.log"
    rc, dur = run_cmd(args, log_path=logp)
    return rc == 0

# ----------------- Config loading -----------------
def load_grid_csv(path: Path):
    rows = []
    with open(path, newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            # helper to safely get and strip a value (returns "" if None)
            def g(key, alt=None):
                v = None
                # case-insensitive lookup: prefer exact key, else lower-case match
                if key in r:
                    v = r.get(key)
                else:
                    # try to find key case-insensitively
                    for k2 in r.keys():
                        if k2 is not None and k2.strip().lower() == key.lower():
                            v = r.get(k2)
                            break
                if v is None:
                    return alt
                # if still None, return alt
                if isinstance(v, str):
                    return v.strip()
                return v

            cfg = {}
            cfg["out_dir"] = g("out_dir") or g("outdir") or g("out") or ""
            cfg["base_data_dir"] = g("base_data_dir") or g("data_dir") or g("base") or "data/synthetic"

            # numeric fields with safe conversion + defaults
            def to_int(val, default=0):
                try:
                    if val is None or str(val).strip() == "":
                        return default
                    return int(float(str(val)))
                except Exception:
                    return default

            def to_float(val, default=0.0):
                try:
                    if val is None or str(val).strip() == "":
                        return default
                    return float(val)
                except Exception:
                    return default

            cfg["n_gene_signal"] = to_int(g("n_gene_signal") or g("n_genes"), 0)
            cfg["n_brain_signal"] = to_int(g("n_brain_signal") or g("n_brain"), 0)
            cfg["n_snp_signal"] = to_int(g("n_snp_signal") or g("n_snps"), 0)
            cfg["effect"] = to_float(g("effect"), 0.0)
            cfg["fraction"] = to_float(g("fraction"), 1.0)
            cfg["seed"] = to_int(g("seed"), 42)

            # boolean-like fields
            def to_bool(val, default=False):
                if val is None:
                    return default
                s = str(val).strip().lower()
                if s == "":
                    return default
                return s in ("1", "true", "yes", "y", "t")

            cfg["copy_all"] = to_bool(g("copy_all"), True)
            cfg["overwrite"] = to_bool(g("overwrite"), False)
            cfg["quick"] = to_bool(g("quick"), False)
            # optional PCA params
            n1 = g("n_gene_pca")
            if n1 and str(n1).strip() != "":
                cfg["n_gene_pca"] = to_int(n1, None)
            n2 = g("n_brain_pca")
            if n2 and str(n2).strip() != "":
                cfg["n_brain_pca"] = to_int(n2, None)
            n3 = g("n_snp_pca")
            if n3 and str(n3).strip() != "":
                cfg["n_snp_pca"] = to_int(n3, None)

            rows.append(cfg)
    return rows

# ----------------- Main entry -----------------
def parse_args():
    p = argparse.ArgumentParser(description="Orchestrate experiments: inject -> train -> explain -> recovery (resume-capable)")
    p.add_argument("--grid-csv", type=str, help="Path to CSV with experiment grid rows. Columns: out_dir,n_gene_signal,n_brain_signal,n_snp_signal,effect,fraction,seed,...")
    p.add_argument("--out-dir", type=str, help="Single experiment output dir (if not using --grid-csv)")
    p.add_argument("--n-gene-signal", type=int, default=50)
    p.add_argument("--n-brain-signal", type=int, default=20)
    p.add_argument("--n-snp-signal", type=int, default=0)
    p.add_argument("--effect", type=float, default=0.8)
    p.add_argument("--fraction", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--base-data-dir", type=str, default="data/synthetic", help="Base synthetic dataset directory")
    p.add_argument("--copy-all", action="store_true", help="Copy full base dataset to out-dir before writing modified arrays")
    p.add_argument("--overwrite", action="store_true", help="Overwrite when copying base dataset")
    p.add_argument("--quick", action="store_true", help="Use quick mode for training (passes --quick to train_fusion_model)")
    p.add_argument("--skip-explain", action="store_true", help="Skip explanation step")
    p.add_argument("--skip-recovery", action="store_true", help="Skip recovery step")
    p.add_argument("--top-explain", type=int, default=200, help="Top features to request from explain script")
    p.add_argument("--top-recovery-k", type=int, default=200, help="Top-K components to inspect in recovery")
    p.add_argument("--top-recovery-n", type=int, default=30, help="Top-N originals per component to inspect in recovery")
    p.add_argument("--log-dir", type=str, default=str(RESULTS_DIR / "logs"))
    return p.parse_args()

def main():
    args = parse_args()
    # prepare log dir and summary path
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = RESULTS_DIR / "experiment_summary.csv"
    # load grid
    configs = []
    if args.grid_csv:
        configs = load_grid_csv(Path(args.grid_csv))
        print(f"[INFO] Loaded {len(configs)} experiments from {args.grid_csv}")
    else:
        if not args.out_dir:
            print("[ERROR] either --grid-csv or --out-dir must be provided")
            return 1
        cfg = {
            "out_dir": args.out_dir,
            "base_data_dir": args.base_data_dir,
            "n_gene_signal": args.n_gene_signal,
            "n_brain_signal": args.n_brain_signal,
            "n_snp_signal": args.n_snp_signal,
            "effect": args.effect,
            "fraction": args.fraction,
            "seed": args.seed,
            "copy_all": args.copy_all,
            "overwrite": args.overwrite,
            "quick": args.quick
        }
        configs = [cfg]

    # ensure summary CSV exists (create header if missing)
    header = ["out_dir","n_gene_signal","n_brain_signal","n_snp_signal","effect","fraction","seed",
              "injection_done","injection_time_s",
              "fusion_done","fusion_time_s",
              "explain_done","explain_time_s",
              "recovery_done","recovery_time_s",
              "fusion_metrics_summary","timestamp"]
    if not summary_csv.exists():
        with open(summary_csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)

    # iterate configs
    for cfg in configs:
        # normalize out_dir to be under project root if relative
        out_dir = Path(cfg["out_dir"])
        if not out_dir.is_absolute():
            out_dir = ROOT / out_dir
        cfg["out_dir"] = str(out_dir)

        print(f"\n====================\nExperiment: {out_dir}\nParams: genes={cfg['n_gene_signal']} brain={cfg['n_brain_signal']} snps={cfg.get('n_snp_signal',0)} effect={cfg['effect']} fraction={cfg['fraction']} seed={cfg['seed']}\n====================")
        # per-experiment log
        base_log = log_dir / f"{out_dir.name}_runner.log"
        with open(base_log, "a", encoding="utf-8") as flog:
            flog.write(f"\n\n===== Experiment start: {time.asctime()} =====\n")
        rec = {
            "out_dir": str(out_dir), "n_gene_signal": cfg["n_gene_signal"], "n_brain_signal": cfg["n_brain_signal"],
            "n_snp_signal": cfg.get("n_snp_signal",0), "effect": cfg["effect"], "fraction": cfg["fraction"], "seed": cfg["seed"],
            "injection_done": False, "injection_time_s": 0.0,
            "fusion_done": False, "fusion_time_s": 0.0,
            "explain_done": False, "explain_time_s": 0.0,
            "recovery_done": False, "recovery_time_s": 0.0,
            "fusion_metrics_summary": None,
            "timestamp": time.time()
        }

        # step 1: injection
        try:
            if injection_done(out_dir):
                print(f"[SKIP] injection already done for {out_dir}")
                rec["injection_done"] = True
            else:
                t0 = time.time()
                ok = run_injection(cfg, base_log)
                rec["injection_time_s"] = round(time.time() - t0, 2)
                rec["injection_done"] = ok
                if not ok:
                    print("[ERROR] injection failed; skipping this experiment")
                    # log and continue to next experiment
                    with open(base_log, "a", encoding="utf-8") as flog:
                        flog.write(f"[ERROR] injection step failed for {out_dir}\n")
                    # still write record and continue to next experiment
                    pass
        except Exception as e:
            print(f"[EXCEPTION] injection step error: {e}")
            with open(base_log, "a", encoding="utf-8") as flog:
                flog.write(f"[EXCEPTION] injection: {e}\n")

        # step 2: fusion training
        try:
            if fusion_training_done():
                print(f"[SKIP] fusion training already done (models+metrics present)")
                rec["fusion_done"] = True
            else:
                t0 = time.time()
                ok = run_fusion(cfg, base_log, quick=args.quick or cfg.get("quick", False))
                rec["fusion_time_s"] = round(time.time() - t0, 2)
                rec["fusion_done"] = ok
                if not ok:
                    print("[ERROR] fusion training failed; continuing to next steps but marking failed")
                    with open(base_log, "a", encoding="utf-8") as flog:
                        flog.write("[ERROR] fusion training failed\n")
        except Exception as e:
            print(f"[EXCEPTION] fusion training error: {e}")
            with open(base_log, "a", encoding="utf-8") as flog:
                flog.write(f"[EXCEPTION] fusion: {e}\n")

        # read metrics if available
        try:
            fm = read_metrics_from_results()
            rec["fusion_metrics_summary"] = json.dumps(fm)
        except Exception:
            rec["fusion_metrics_summary"] = None

        # step 3: explanation
        if args.skip_explain:
            print("[SKIP] Explanation step requested to skip")
        else:
            try:
                if explanation_done():
                    print("[SKIP] explanation already done")
                    rec["explain_done"] = True
                else:
                    t0 = time.time()
                    ok = run_explain(cfg, base_log, top=args.top_explain, force_interventional=True)
                    rec["explain_time_s"] = round(time.time() - t0, 2)
                    rec["explain_done"] = ok
            except Exception as e:
                print(f"[EXCEPTION] explain error: {e}")
                with open(base_log, "a", encoding="utf-8") as flog:
                    flog.write(f"[EXCEPTION] explain: {e}\n")

        # step 4: recovery
        if args.skip_recovery:
            print("[SKIP] Recovery step requested to skip")
        else:
            try:
                if recovery_done():
                    print("[SKIP] recovery already done")
                    rec["recovery_done"] = True
                else:
                    t0 = time.time()
                    ok = run_recovery(cfg, base_log, top_k=args.top_recovery_k, top_n=args.top_recovery_n)
                    rec["recovery_time_s"] = round(time.time() - t0, 2)
                    rec["recovery_done"] = ok
            except Exception as e:
                print(f"[EXCEPTION] recovery error: {e}")
                with open(base_log, "a", encoding="utf-8") as flog:
                    flog.write(f"[EXCEPTION] recovery: {e}\n")

        # finalize rec timestamp and append to experiment_summary.csv
        rec["timestamp"] = time.time()
        # append
        with open(RESULTS_DIR / "experiment_summary.csv", "a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                rec["out_dir"], rec["n_gene_signal"], rec["n_brain_signal"], rec["n_snp_signal"],
                rec["effect"], rec["fraction"], rec["seed"],
                int(rec["injection_done"]), rec["injection_time_s"],
                int(rec["fusion_done"]), rec["fusion_time_s"],
                int(rec["explain_done"]), rec["explain_time_s"],
                int(rec["recovery_done"]), rec["recovery_time_s"],
                rec["fusion_metrics_summary"], rec["timestamp"]
            ])
        print(f"[SUMMARY] Experiment record written for {out_dir}")

    print("\nAll experiments finished (or attempted). See results/experiment_summary.csv and logs in", log_dir)
    return 0

if __name__ == "__main__":
    import argparse
    exit(main())

#!/usr/bin/env python3
"""
run_full_pipeline.py (Improved Version)

One-click orchestrator to run the full Schizophrenia ML project pipeline
with progress tracking, targeted gene mapping support, and detailed logging.

Steps:
  1. Inject synthetic or targeted signal
  2. Inject synthetic noise (optional)
  3. Train gene-only models
  4. Train brain-only models
  5. Train fusion multimodal model
  6. Explain fusion model
  7. Recovery analysis
  8. Visualization
  9. Final consolidated report

Usage:
    python scripts/run_full_pipeline.py --data-dir data/synthetic --out-dir data/synthetic_fullrun --effect 1.5 --quick
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

# ---------------------------------------------------------------------------
# Helper: run one command with progress indicator
# ---------------------------------------------------------------------------
def run_cmd(cmd, step_name):
    """Run a subprocess command with progress spinner and timing."""
    console.print(f"[bold cyan]\n⏳ {step_name} started...\n")
    with Progress(
        SpinnerColumn(),
        TextColumn(f"[bold blue]{step_name}"),
        transient=True,
        console=console
    ) as progress:
        task = progress.add_task("run", start=False)
        t0 = time.time()
        rc = subprocess.call(cmd, shell=False)
        dt = time.time() - t0

    if rc == 0:
        console.print(f"[green]✅ {step_name} completed in {dt:.1f}s\n")
        return dt
    else:
        console.print(f"[red]❌ {step_name} failed (rc={rc}) — aborting pipeline.\n")
        sys.exit(1)

# ---------------------------------------------------------------------------
# Main pipeline logic
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="data/synthetic", help="Base dataset directory")
    p.add_argument("--out-dir", type=str, default="data/synthetic_fullrun", help="Output dataset directory")
    p.add_argument("--n-gene-signal", type=int, default=20)
    p.add_argument("--n-brain-signal", type=int, default=10)
    p.add_argument("--n-snp-signal", type=int, default=0)
    p.add_argument("--effect", type=float, default=1.5)
    p.add_argument("--fraction", type=float, default=0.8)
    p.add_argument("--mislabel-fraction", type=float, default=0.0)
    p.add_argument("--missing-fraction", type=float, default=0.0)
    p.add_argument("--feature-drop-fraction", type=float, default=0.0)
    p.add_argument("--gaussian-noise-sd", type=float, default=0.0)
    p.add_argument("--quick", action="store_true", help="Quick mode for fast testing (less data, fewer folds)")
    args = p.parse_args()

    scripts = Path(__file__).resolve().parents[0]
    total_start = time.time()

    console.rule("[bold yellow]🚀 Starting Full Schizophrenia ML Pipeline")

    durations = {}

    # --- Step 1: Inject Signal (Targeted if gene map exists) ---
    gene_map_path = Path("results/gene_symbol_mapping.csv")
    inject_cmd = [
        sys.executable, str(scripts / "inject_signal.py"),
        "--data-dir", args.data_dir,
        "--out-dir", args.out_dir,
        "--n-gene-signal", str(args.n_gene_signal),
        "--n-brain-signal", str(args.n_brain_signal),
        "--n-snp-signal", str(args.n_snp_signal),
        "--effect", str(args.effect),
        "--fraction", str(args.fraction),
        "--copy-all", "--overwrite"
    ]

    if gene_map_path.exists():
        console.print("[green]🧬 Detected gene_symbol_mapping.csv — using targeted injection mode.")
        inject_cmd.insert(4, "--mode")
        inject_cmd.insert(5, "targeted")
        inject_cmd.insert(6, "--gene-map")
        inject_cmd.insert(7, str(gene_map_path))

    durations["Step 1: Inject Signal"] = run_cmd(inject_cmd, "Step 1/9: Inject synthetic or targeted signal")

    # --- Step 2: Inject Noise ---
    if any([args.mislabel_fraction, args.missing_fraction, args.feature_drop_fraction, args.gaussian_noise_sd]):
        durations["Step 2: Inject Noise"] = run_cmd([
            sys.executable, str(scripts / "inject_noise.py"),
            "--data-dir", args.out_dir,
            "--out-dir", args.out_dir,
            "--copy-all", "--overwrite",
            "--mislabel-fraction", str(args.mislabel_fraction),
            "--missing-fraction", str(args.missing_fraction),
            "--feature-drop-fraction", str(args.feature_drop_fraction),
            "--gaussian-noise-sd", str(args.gaussian_noise_sd)
        ], "Step 2/9: Inject synthetic noise")
    else:
        console.print("[blue]ℹ️ Step 2/9 skipped — no noise requested.\n")

    # --- Step 3: Train Gene Models ---
    durations["Step 3: Gene Models"] = run_cmd([
        sys.executable, str(scripts / "train_gene_model.py"),
        "--data-dir", args.out_dir
    ] + (["--quick"] if args.quick else []), "Step 3/9: Train gene-only models")

    # --- Step 4: Train Brain Models ---
    durations["Step 4: Brain Models"] = run_cmd([
        sys.executable, str(scripts / "train_brain_model.py"),
        "--data-dir", args.out_dir
    ] + (["--quick"] if args.quick else []), "Step 4/9: Train brain-only models")

    # --- Step 5: Train Fusion Model ---
    durations["Step 5: Fusion Model"] = run_cmd([
        sys.executable, str(scripts / "train_fusion_model.py"),
        "--data-dir", args.out_dir
    ] + (["--quick"] if args.quick else []), "Step 5/9: Train fusion multimodal model")

    # --- Step 6: Explain Fusion Model ---
    durations["Step 6: Explain Fusion Model"] = run_cmd([
        sys.executable, str(scripts / "explain_fusion_model.py"),
        "--model", "models/fusion_rf.pkl",
        "--data-dir", args.out_dir,
        "--top", "100", "--force-interventional"
    ], "Step 6/9: Explain fusion model")

    # --- Step 7: Recovery Analysis ---
    durations["Step 7: Recovery"] = run_cmd([
        sys.executable, str(scripts / "check_recovery.py"),
        "--data-dir", args.out_dir,
        "--top-k", "200",
        "--top-n", "30"
    ], "Step 7/9: Recovery analysis")

    # --- Step 8: Visualization ---
    durations["Step 8: Visualization"] = run_cmd([
        sys.executable, str(scripts / "visualize_results.py"),
    ], "Step 8/9: Visualization")

    # --- Step 9: Final Report ---
    durations["Step 9: Final Report"] = run_cmd([
        sys.executable, str(scripts / "final_report.py"),
    ], "Step 9/9: Final consolidated report")

    # --- Summary ---
    total_time = time.time() - total_start
    console.rule("[bold green]🎉 Pipeline Complete")
    console.print(f"[bold]All results saved in:[/bold] {Path('results').resolve()}")
    console.print(f"[bold]Final report saved in:[/bold] {Path('results/final_report').resolve()}\n")

    console.print("[bold yellow]⏱ Step-wise duration summary:")
    for step, dur in durations.items():
        console.print(f"   - {step:<30} : {dur:.1f}s")
    console.print(f"\n[bold cyan]Total pipeline runtime:[/bold cyan] {total_time/60:.1f} min\n")

if __name__ == "__main__":
    main()

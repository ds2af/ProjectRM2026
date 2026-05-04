"""
scripts/train_all.py
====================
Master training script: trains all models sequentially on GPU, records
wall-clock training time per model, then runs evaluation, figures, and VTK export.

AI Disclaimer
-------------
Coding assistance from ChatGPT and GitHub Copilot was used during development.
The author has thoroughly reviewed, checked, and verified the code for correctness
and takes responsibility for the final implementation used in this project.

Usage
-----
    python scripts/train_all.py --config configs/default.yaml
    python scripts/train_all.py --config configs/default.yaml --quick  # sanity check

Output summary saved to:  results/training_summary.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
    sys.stderr.reconfigure(line_buffering=True, write_through=True)
except Exception:
    pass


def parse_args():
    p = argparse.ArgumentParser(description="Train all surrogate models sequentially")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--quick", action="store_true", help="Quick mode for all models")
    p.add_argument("--epochs", type=int, default=None, help="Override training epochs for all model scripts")
    p.add_argument("--skip_vtk", action="store_true", help="Skip VTK export step")
    p.add_argument("--models", nargs="+",
                   default=["unet", "unet_lomix", "fno"],
                   help="Subset of models to train (unet, unet_lomix, fno)")
    return p.parse_args()


def run_script(script: str, extra_args: list[str], label: str) -> tuple[int, float]:
    """Run a script as subprocess, return (exit_code, elapsed_seconds)."""
    cmd = [sys.executable, "-u", str(ROOT / "scripts" / script)] + extra_args
    print(f"\n{'='*70}")
    print(f"  STARTING: {label}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"  Time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*70}")

    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.perf_counter() - t0

    status = "✅ OK" if result.returncode == 0 else f"❌ FAILED (exit {result.returncode})"
    print(f"\n  {status}  |  {label}  |  {elapsed/60:.1f} min ({elapsed:.0f}s)")
    return result.returncode, elapsed


def fmt_time(s: float) -> str:
    h, m = int(s // 3600), int((s % 3600) // 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {int(s%60):02d}s"


def main():
    args = parse_args()
    base_args = ["--config", args.config] + (["--quick"] if args.quick else [])
    if args.epochs is not None:
        base_args += ["--epochs", str(args.epochs)]

    summary = {
        "run_date": datetime.now().isoformat(),
        "config": args.config,
        "quick_mode": args.quick,
        "models": {},
    }

    # ── Map model name → training script + extra flags ─────────────────
    MODEL_SCRIPTS = {
        "unet": ("train_unet.py", ["--vanilla"]),
        "unet_lomix": ("train_unet.py", []),  # default uses LoMix
        "fno": ("train_fno.py", []),
    }

    total_start = time.perf_counter()

    # ── Train each model ────────────────────────────────────────────────
    for model in args.models:
        if model not in MODEL_SCRIPTS:
            print(f"[train_all] Unknown model '{model}', skipping.")
            continue

        script, extra = MODEL_SCRIPTS[model]
        rc, elapsed = run_script(script, base_args + extra, model.upper())

        summary["models"][model] = {
            "exit_code":      rc,
            "train_time_s":   round(elapsed, 1),
            "train_time_fmt": fmt_time(elapsed),
            "status":         "ok" if rc == 0 else "failed",
        }

    total_elapsed = time.perf_counter() - total_start
    summary["total_train_time_s"]   = round(total_elapsed, 1)
    summary["total_train_time_fmt"] = fmt_time(total_elapsed)

    # ── Aggregate metrics ───────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  POST-TRAINING: Aggregating metrics ...")
    rc_eval, _ = run_script("evaluate_all.py", [], "evaluate_all")
    summary["evaluate_exit_code"] = rc_eval

    # Attach inference times and accuracy from metrics_summary.json
    metrics_path = ROOT / "results" / "metrics_summary.json"
    if metrics_path.exists():
        with metrics_path.open() as f:
            ms = json.load(f)
        for model, mdata in ms.items():
            if model in summary["models"]:
                summary["models"][model]["rmse"]             = mdata.get("rmse")
                summary["models"][model]["inference_time_s"] = mdata.get("inference_time_s")
                summary["models"][model]["n_params"]         = mdata.get("n_params")
                summary["models"][model]["source"]           = mdata.get("source", "measured")

    # ── Save training summary ───────────────────────────────────────────
    out_path = ROOT / "results" / "training_summary.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[train_all] Training summary saved → {out_path}")

    # ── Generate figures ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  POST-TRAINING: Generating figures ...")
    rc_fig, _ = run_script("generate_figures.py", [], "generate_figures")
    summary["figures_exit_code"] = rc_fig

    # ── Export VTK ──────────────────────────────────────────────────────
    vtk_script = ROOT / "scripts" / "export_vtk.py"
    if not args.skip_vtk and vtk_script.exists():
        print(f"\n{'='*70}")
        print("  POST-TRAINING: Exporting VTK files ...")
        rc_vtk, elapsed_vtk = run_script(
            "export_vtk.py",
            ["--model", "all", "--n_samples", "1", "--n_timesteps", "100"],
            "export_vtk",
        )
        summary["vtk_exit_code"]    = rc_vtk
        summary["vtk_time_s"]       = round(elapsed_vtk, 1)
    elif not args.skip_vtk:
        print("\n[train_all] Skipping VTK export (scripts/export_vtk.py not found).")

    # ── Final report ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  COMPLETE — total wall clock: {fmt_time(total_elapsed)}")
    print(f"{'='*70}")
    print(f"\n  {'Model':<14} {'Train time':>12} {'RMSE':>12} {'Infer(s)':>10} {'Params':>12}")
    print(f"  {'-'*60}")
    for model, info in summary["models"].items():
        rmse_s  = f"{info.get('rmse', 'N/A'):.4e}"   if isinstance(info.get('rmse'), float)  else "N/A"
        inf_s   = f"{info.get('inference_time_s', 0):.3f}" if info.get('inference_time_s') else "N/A"
        par_s   = f"{info.get('n_params', 0):,}"     if info.get('n_params')               else "N/A"
        print(f"  {model:<14} {info['train_time_fmt']:>12} {rmse_s:>12} {inf_s:>10} {par_s:>12}")
    print()

    # Update summary with final timing
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

"""
scripts/run_repeated_significance.py
=====================================
Run repeated training and generate statistical significance analysis.

For each repeat, trains unet, unet_lomix, and fno models with fixed seeds,
collects metrics, and generates box-and-whisker plots.

Usage
-----
    python scripts/run_repeated_significance.py --config configs/default.yaml
    python scripts/run_repeated_significance.py --config configs/default.yaml --repeats 5 --epochs 200
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description="Repeated experiments for significance")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed-start", type=int, default=100)
    p.add_argument("--epochs", type=int, default=50, help="Epochs per model training")
    p.add_argument("--out-dir", default="results/runtime_significance")
    return p.parse_args()


def run_model_training(model_name: str, seed: int, epochs: int, config: str) -> dict | None:
    """Train a single model and return metrics."""
    if model_name in ["unet", "unet_lomix"]:
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "train_unet.py"),
            "--config", config,
            "--seed", str(seed),
            "--epochs", str(epochs),
        ]
        if model_name == "unet":
            cmd.append("--vanilla")
    elif model_name == "fno":
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "train_fno.py"),
            "--config", config,
            "--seed", str(seed),
            "--epochs", str(epochs),
        ]
    else:
        return None
    
    try:
        subprocess.run(cmd, check=True, cwd=ROOT)
        # Load metrics
        metrics_path = ROOT / "results" / f"{model_name}_metrics.json"
        if metrics_path.exists():
            with metrics_path.open() as f:
                return json.load(f)
    except subprocess.CalledProcessError as e:
        print(f"[warn] Training failed for {model_name}: {e}")
    return None


def main():
    args = parse_args()
    models = ["unet", "unet_lomix", "fno"]
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = out_dir / "repeated_runs.csv"
    existing_rows = []
    if csv_path.exists():
        with csv_path.open("r") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader) if reader.fieldnames else []
    
    print(f"[run_repeated_significance] Running {args.repeats} repeats with {len(models)} models...")
    print(f"Output: {csv_path}")
    
    results = []
    for repeat in range(args.repeats):
        seed = args.seed_start + repeat
        print(f"\n[repeat {repeat+1}/{args.repeats}] Seed: {seed}")
        
        for model_name in models:
            print(f"  Training {model_name}...")
            metrics = run_model_training(model_name, seed, args.epochs, args.config)
            
            if metrics:
                results.append({
                    "repeat": repeat,
                    "seed": seed,
                    "model": model_name,
                    "rmse": metrics.get("rmse", float("nan")),
                    "rel_l2": metrics.get("rel_l2", float("nan")),
                    "max_error": metrics.get("max_error", float("nan")),
                    "inference_time_s": metrics.get("inference_time_s", float("nan")),
                    "n_params": metrics.get("n_params", 0),
                })
    
    # Save results
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["repeat", "seed", "model", "rmse", "rel_l2", "max_error", "inference_time_s", "n_params"])
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    
    print(f"\n[run_repeated_significance] Results saved to {csv_path}")
    print(f"Total runs completed: {len(results)}")


if __name__ == "__main__":
    main()

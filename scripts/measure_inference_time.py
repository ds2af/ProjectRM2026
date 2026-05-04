"""
Generate inference time boxplot with bootstrap 95% confidence intervals.

Can either:
1. Generate synthetic inference time data for visualization testing
2. Load existing inference_times.csv and create boxplot

AI Disclaimer
-------------
Coding assistance from ChatGPT and GitHub Copilot was used during development.
The author has thoroughly reviewed, checked, and verified the code for correctness
and takes responsibility for the final implementation used in this project.

Usage
-----
python scripts/measure_inference_time.py --generate-demo
python scripts/measure_inference_time.py --csv-file results/inference_timing/inference_times.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate boxplot with bootstrap 95% CI for inference times")
    p.add_argument("--generate-demo", action="store_true", help="Generate synthetic demo data")
    p.add_argument("--csv-file", type=str, default=None, help="Path to inference_times.csv")
    p.add_argument("--out-dir", default="results/inference_timing", help="Output directory for plot")
    p.add_argument("--runs", type=int, default=25, help="Number of runs per model (for demo)")
    return p.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def generate_demo_data(num_runs: int = 25) -> Dict[str, List[float]]:
    """Generate synthetic inference time data for demo/testing."""
    np.random.seed(42)
    
    # Realistic inference times (seconds) with some variance
    # UNet: slower
    unet_times = np.random.normal(loc=1.50, scale=0.08, size=num_runs).clip(1.2, 2.0)
    # UNet-LoMix: slightly slower due to multi-scale fusion
    unet_lomix_times = np.random.normal(loc=1.65, scale=0.10, size=num_runs).clip(1.3, 2.1)
    # FNO: faster (spectral efficiency)
    fno_times = np.random.normal(loc=0.95, scale=0.06, size=num_runs).clip(0.7, 1.2)
    
    return {
        "U-Net": unet_times.tolist(),
        "U-Net (LoMix)": unet_lomix_times.tolist(),
        "FNO": fno_times.tolist(),
    }


def load_csv_data(csv_path: Path) -> Dict[str, List[float]]:
    """Load inference times from CSV file."""
    inference_times = {}
    
    try:
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                model = row["model"]
                time_s = float(row["inference_time_s"])
                
                if model not in inference_times:
                    inference_times[model] = []
                inference_times[model].append(time_s)
        
        print(f"[info] Loaded {len(inference_times)} models from {csv_path}")
        for model, times in inference_times.items():
            print(f"  {model}: {len(times)} runs")
        
        return inference_times
    
    except Exception as e:
        print(f"[error] Failed to load CSV: {e}")
        return {}


def generate_boxplot(
    inference_times: Dict[str, List[float]],
    out_dir: Path,
    num_runs: int,
) -> None:
    """Generate boxplot with bootstrap 95% CI notches."""

    models = ["U-Net", "U-Net (LoMix)", "FNO"]
    data = [inference_times.get(m, []) for m in models]

    # Filter out empty lists
    valid_models = [m for m, times in zip(models, data) if times]
    valid_data = [times for times in data if times]

    if not valid_data:
        print("[warn] No data available for plotting")
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    box_kwargs = dict(
        notch=True,
        bootstrap=10000,
        showmeans=True,
        meanline=False,
        whis=1.5,
        medianprops={"linewidth": 2.0, "color": "#222"},
        whiskerprops={"linewidth": 1.4, "color": "#555"},
        capprops={"linewidth": 1.4, "color": "#555"},
        boxprops={"linewidth": 1.2, "color": "#444"},
    )

    try:
        box = ax.boxplot(valid_data, labels=valid_models, **box_kwargs)
    except TypeError:
        box = ax.boxplot(valid_data, tick_labels=valid_models, **box_kwargs)

    palette = ["#4C78A8", "#F58518", "#54A24B"]
    for patch, color in zip(box["boxes"], palette[: len(valid_models)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.28)

    # Overlay scatter points
    rng = np.random.default_rng(42)
    for i, y in enumerate(valid_data, start=1):
        x = rng.normal(i, 0.045, size=len(y))
        ax.scatter(
            x, y, s=28, alpha=0.75, color=palette[i - 1], edgecolors="white", linewidths=0.4
        )

    ax.set_title(f"Inference Runtime Distribution ({num_runs} Runs per Model)", fontsize=13)
    ax.set_xlabel("Model")
    ax.set_ylabel("Inference Time (seconds)")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    fig.text(
        0.5,
        0.01,
        "Notches indicate bootstrap 95% confidence intervals for the median.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#555555",
    )

    fig.tight_layout()
    out_path = out_dir / "inference_time_boxplot.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[info] Boxplot saved to {out_path}")


def main() -> None:
    args = parse_args()
    out_dir = ROOT / args.out_dir
    ensure_dir(out_dir)

    print(f"[info] Output dir: {out_dir}")

    # Load or generate data
    if args.generate_demo:
        print(f"\n=== Generating demo data ({args.runs} runs per model) ===")
        inference_times = generate_demo_data(args.runs)
        num_runs = args.runs
    elif args.csv_file:
        print(f"\n=== Loading data from CSV ===")
        csv_path = Path(args.csv_file)
        if not csv_path.exists():
            print(f"[error] CSV file not found: {csv_path}")
            return
        inference_times = load_csv_data(csv_path)
        # Infer number of runs per model
        if inference_times:
            num_runs = len(next(iter(inference_times.values())))
        else:
            num_runs = 0
    else:
        print("[error] Provide either --generate-demo or --csv-file")
        return

    if not inference_times:
        print("[error] No data loaded")
        return

    # Print summary
    print(f"\n=== Data Summary ===")
    for model, times in inference_times.items():
        if times:
            mean_t = np.mean(times)
            std_t = np.std(times)
            min_t = np.min(times)
            max_t = np.max(times)
            print(f"{model:18s}: mean={mean_t:.4f}s, std={std_t:.6f}s, min={min_t:.4f}s, max={max_t:.4f}s")

    # Generate boxplot
    print(f"\n=== Generating boxplot ===")
    generate_boxplot(inference_times, out_dir, num_runs)


if __name__ == "__main__":
    main()

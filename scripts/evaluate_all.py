"""
scripts/evaluate_all.py
=======================
Aggregate measured metrics from trained models into results/metrics_summary.json.

This script is strict by design:
- Missing model files are reported as source="missing" with NaN metrics.
- Only includes the three models: unet, unet_lomix, fno

Usage
-----
    python scripts/evaluate_all.py
    python scripts/evaluate_all.py --results_dir results
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL_ORDER = ["unet", "unet_lomix", "fno"]
MODEL_DISPLAY_NAMES = {
    "unet": "U-Net",
    "unet_lomix": "U-Net (LoMix)",
    "fno": "FNO",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results")
    return p.parse_args()


def _nan() -> float:
    return float("nan")


def load_metric(results_dir: Path, model_key: str) -> dict:
    path = results_dir / f"{model_key}_metrics.json"
    if not path.exists():
        print(f"  [missing] {path.name}")
        return {
            "model": model_key,
            "rmse": _nan(),
            "inference_time_s": _nan(),
            "n_params": 0,
            "source": "missing",
        }

    with path.open() as f:
        data = json.load(f)

    print(f"  [ok] {path.name}")
    data["source"] = "measured"
    return data


def _is_finite(v) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(v)


def main():
    args = parse_args()
    res_dir = ROOT / args.results_dir
    res_dir.mkdir(parents=True, exist_ok=True)

    print("\n[evaluate_all] Collecting measured model metrics ...")
    summary: dict[str, dict] = {}

    for key in MODEL_ORDER:
        metrics = load_metric(res_dir, key)
        summary[key] = {
            "display_name": MODEL_DISPLAY_NAMES[key],
            "rmse": metrics.get("rmse", _nan()),
            "inference_time_s": metrics.get("inference_time_s", _nan()),
            "n_params": metrics.get("n_params", 0),
            "source": metrics.get("source", "missing"),
        }

    out_path = res_dir / "metrics_summary.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print("\n[evaluate_all] ----- Results Summary -----")
    print(f"{'Model':<16}{'RMSE':>12}{'Inf.time(s)':>14}{'Source':>12}")
    print("-" * 56)
    for key in MODEL_ORDER:
        row = summary[key]
        rmse = row["rmse"]
        t = row["inference_time_s"]
        rmse_s = f"{rmse:.4e}" if _is_finite(rmse) else "nan"
        t_s = f"{t:.3f}" if _is_finite(t) else "nan"
        print(f"{row['display_name']:<16}{rmse_s:>12}{t_s:>14}{row['source']:>12}")

    print(f"\n[evaluate_all] Summary saved -> {out_path}")


if __name__ == "__main__":
    main()

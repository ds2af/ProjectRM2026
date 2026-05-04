"""
Run repeated training experiments and statistical significance analysis.

AI Disclaimer
-------------
Coding assistance from ChatGPT and GitHub Copilot was used during development.
The author has thoroughly reviewed, checked, and verified the code for correctness
and takes responsibility for the final implementation used in this project.


Models:
- UNet (vanilla)
- UNet (LoMix)
- FNO

For each repeat (default 10), this script trains all four models with a fixed
seed, collects RMSE and inference time, computes omnibus + pairwise significance,
and generates a box-and-whisker plot with significance annotations.

Usage
-----
python scripts/run_repeated_significance.py --config configs/default.yaml
python scripts/run_repeated_significance.py --config configs/default.yaml --repeats 10 --epochs 200
python scripts/run_repeated_significance.py --config configs/default.yaml --quick --repeats 10
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    from scipy import stats  # type: ignore[reportMissingImports]
except Exception:  # pragma: no cover
    stats = None

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Repeated experiments + significance for UNet/UNet-LoMix/FNO")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--seed-start", type=int, default=100)
    p.add_argument("--epochs", type=int, default=500, help="Epochs per model training (default: 500)")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--metric", default="rmse", choices=["rmse", "inference_time_s"])
    p.add_argument("--out-dir", default="results/statistical_significance")
    p.add_argument("--max-retries", type=int, default=1, help="Retries per failed model run")
    p.add_argument("--resume-existing", action="store_true", help="Resume from existing repeated_runs.csv if present")
    return p.parse_args()


def run_cmd(cmd: List[str]) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def run_cmd_with_retry(cmd: List[str], resume_cmd: List[str], max_retries: int) -> None:
    for attempt in range(max_retries + 1):
        active_cmd = cmd if attempt == 0 else resume_cmd
        try:
            run_cmd(active_cmd)
            return
        except subprocess.CalledProcessError as exc:
            if attempt >= max_retries:
                raise
            print(
                f"[warn] Command failed with exit code {exc.returncode}. "
                f"Retrying ({attempt + 1}/{max_retries}) with resume..."
            )


def load_existing_rows(csv_path: Path) -> List[dict]:
    if not csv_path.exists():
        return []

    rows: List[dict] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "repeat": int(r["repeat"]),
                    "seed": int(r["seed"]),
                    "model": r["model"],
                    "rmse": float(r["rmse"]),
                    "inference_time_s": float(r["inference_time_s"]),
                    "n_params": float(r["n_params"]),
                }
            )
    return rows


def append_row(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = (not csv_path.exists()) or csv_path.stat().st_size == 0
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["repeat", "seed", "model", "rmse", "inference_time_s", "n_params"])
        writer.writerow(
            [
                row["repeat"],
                row["seed"],
                row["model"],
                f"{row['rmse']:.10e}",
                f"{row['inference_time_s']:.10e}",
                f"{row['n_params']:.0f}",
            ]
        )


def load_metrics(model_name: str) -> Dict[str, float]:
    path = ROOT / "results" / f"{model_name}_metrics.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Cliff's delta effect size."""
    gt = 0
    lt = 0
    for x in a:
        gt += np.sum(x > b)
        lt += np.sum(x < b)
    n = a.size * b.size
    if n == 0:
        return float("nan")
    return float((gt - lt) / n)


def holm_adjust(pvals: List[float]) -> List[float]:
    """Holm-Bonferroni adjusted p-values."""
    m = len(pvals)
    order = np.argsort(pvals)
    p_sorted = [pvals[i] for i in order]

    adj_sorted = [0.0] * m
    running_max = 0.0
    for i, p in enumerate(p_sorted):
        adj = (m - i) * p
        running_max = max(running_max, adj)
        adj_sorted[i] = min(1.0, running_max)

    adj = [0.0] * m
    for idx, orig_idx in enumerate(order):
        adj[orig_idx] = adj_sorted[idx]
    return adj


def p_to_star(p: float) -> str:
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def make_plot(
    out_path: Path,
    metric: str,
    labels: List[str],
    data: List[np.ndarray],
    best_label: str,
    pairwise_rows: List[dict],
) -> bool:
    try:
        import matplotlib.pyplot as plt  # Local import so training can proceed if plotting backend is broken
    except Exception as exc:
        print(f"[warn] Skipping plot generation: matplotlib import failed: {exc}")
        return False

    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=180)

    box_kwargs = dict(
        patch_artist=True,
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
        box = ax.boxplot(data, tick_labels=labels, **box_kwargs)
    except TypeError:
        box = ax.boxplot(data, labels=labels, **box_kwargs)

    palette = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
    for patch, color in zip(box["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.28)

    rng = np.random.default_rng(42)
    for i, y in enumerate(data, start=1):
        x = rng.normal(i, 0.045, size=len(y))
        ax.scatter(x, y, s=28, alpha=0.75, color=palette[i - 1], edgecolors="white", linewidths=0.4)

    ax.set_title(f"{metric.upper()} Distribution Across 10 Repeated Runs", fontsize=13)
    ax.set_xlabel("Model")
    ax.set_ylabel(metric)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    max_y = max(float(np.max(arr)) for arr in data)
    min_y = min(float(np.min(arr)) for arr in data)
    span = max(1e-12, max_y - min_y)
    base = max_y + 0.08 * span
    step = 0.08 * span

    idx_by_label = {lab: i + 1 for i, lab in enumerate(labels)}
    anchor_rows = [r for r in pairwise_rows if best_label in (r["model_a"], r["model_b"]) and r["p_holm"] is not None]

    for k, row in enumerate(anchor_rows):
        a = row["model_a"]
        b = row["model_b"]
        x1, x2 = idx_by_label[a], idx_by_label[b]
        if x1 > x2:
            x1, x2 = x2, x1

        y = base + k * step
        ax.plot([x1, x1, x2, x2], [y, y + 0.02 * span, y + 0.02 * span, y], color="#333", linewidth=1.2)
        star = p_to_star(float(row["p_holm"]))
        txt = f"{star} (p={row['p_holm']:.3g})"
        ax.text((x1 + x2) / 2, y + 0.025 * span, txt, ha="center", va="bottom", fontsize=9)

    ax.text(
        0.99,
        0.02,
        f"Best median: {best_label}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#333",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "#f7f7f7", "edgecolor": "#bbb", "alpha": 0.9},
    )
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
    fig.savefig(out_path)
    plt.close(fig)
    return True


def main() -> None:
    args = parse_args()
    out_dir = ROOT / args.out_dir
    ensure_dir(out_dir)

    csv_path = out_dir / "repeated_runs.csv"

    models = [
        {"label": "UNet", "metrics_key": "unet", "cmd": ["scripts/train_unet.py", "--vanilla"], "supports_resume": True},
        {"label": "UNet-LoMix", "metrics_key": "unet_lomix", "cmd": ["scripts/train_unet.py"], "supports_resume": True},
        {"label": "FNO", "metrics_key": "fno", "cmd": ["scripts/train_fno.py"], "supports_resume": True},
    ]

    all_rows: List[dict] = load_existing_rows(csv_path) if args.resume_existing else []
    completed = {(r["repeat"], r["model"]) for r in all_rows}
    if all_rows:
        print(f"[info] Loaded {len(all_rows)} completed runs from {csv_path}")

    for rep in range(args.repeats):
        seed = args.seed_start + rep
        print(f"\n=== Repeat {rep + 1}/{args.repeats} | seed={seed} ===")

        for m in models:
            run_key = (rep + 1, m["label"])
            if run_key in completed:
                print(f"[skip] Repeat {rep + 1} {m['label']} already completed in CSV")
                continue

            cmd = [sys.executable, *m["cmd"], "--config", args.config, "--seed", str(seed)]
            if args.quick:
                cmd.append("--quick")
            else:
                # Only pass epochs if not in quick mode (quick mode has its own epoch count)
                cmd.extend(["--epochs", str(args.epochs)])

            resume_cmd = list(cmd)
            if m.get("supports_resume", False):
                resume_cmd.append("--resume")

            run_cmd_with_retry(cmd, resume_cmd, max_retries=max(0, int(args.max_retries)))
            metrics = load_metrics(m["metrics_key"])
            row = {
                "repeat": rep + 1,
                "seed": seed,
                "model": m["label"],
                "rmse": float(metrics.get("rmse", math.nan)),
                "inference_time_s": float(metrics.get("inference_time_s", math.nan)),
                "n_params": float(metrics.get("n_params", math.nan)),
            }
            all_rows.append(row)
            completed.add(run_key)
            append_row(csv_path, row)
            print(f"  -> {m['label']}: rmse={row['rmse']:.4e}, t_inf={row['inference_time_s']:.4f}s")

    metric = args.metric
    by_model: Dict[str, np.ndarray] = {}
    for m in models:
        vals = [r[metric] for r in all_rows if r["model"] == m["label"]]
        by_model[m["label"]] = np.asarray(vals, dtype=float)

    summary = {}
    for label, arr in by_model.items():
        summary[label] = {
            "n": int(arr.size),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
            "median": float(np.median(arr)),
            "q1": float(np.percentile(arr, 25)),
            "q3": float(np.percentile(arr, 75)),
        }

    labels = [m["label"] for m in models]
    groups = [by_model[label] for label in labels]

    omnibus = {"test": None, "statistic": None, "p_value": None}
    pairwise_rows: List[dict] = []

    if stats is not None:
        kw = stats.kruskal(*groups)
        omnibus = {"test": "kruskal", "statistic": float(kw.statistic), "p_value": float(kw.pvalue)}

        pairs = list(itertools.combinations(labels, 2))
        raw_ps = []
        raw_rows = []
        for a, b in pairs:
            x = by_model[a]
            y = by_model[b]
            mw = stats.mannwhitneyu(x, y, alternative="two-sided")
            p = float(mw.pvalue)
            raw_ps.append(p)
            raw_rows.append(
                {
                    "model_a": a,
                    "model_b": b,
                    "test": "mannwhitneyu",
                    "u_stat": float(mw.statistic),
                    "p_raw": p,
                    "p_holm": None,
                    "cliffs_delta": cliffs_delta(x, y),
                }
            )

        adj_ps = holm_adjust(raw_ps)
        for row, p_adj in zip(raw_rows, adj_ps):
            row["p_holm"] = float(p_adj)
            pairwise_rows.append(row)
    else:
        print("[warn] scipy is unavailable; significance tests were skipped.")

    best_label = min(summary.keys(), key=lambda k: summary[k]["median"])

    plot_path = out_dir / f"boxplot_{metric}.png"
    plot_ok = make_plot(
        out_path=plot_path,
        metric=metric,
        labels=labels,
        data=[by_model[label] for label in labels],
        best_label=best_label,
        pairwise_rows=pairwise_rows,
    )

    out_payload = {
        "config": args.config,
        "repeats": args.repeats,
        "seed_start": args.seed_start,
        "metric_for_significance": metric,
        "summary": summary,
        "omnibus": omnibus,
        "pairwise": pairwise_rows,
        "artifacts": {
            "runs_csv": str(csv_path.relative_to(ROOT)),
            "boxplot": str(plot_path.relative_to(ROOT)) if plot_ok else None,
        },
    }
    summary_path = out_dir / "significance_summary.json"
    write_json(summary_path, out_payload)

    print("\n=== Done ===")
    print(f"Runs CSV: {csv_path}")
    print(f"Summary JSON: {summary_path}")
    print(f"Box plot: {plot_path}")


if __name__ == "__main__":
    main()

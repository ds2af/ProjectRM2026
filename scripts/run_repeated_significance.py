"""
scripts/run_repeated_significance.py
====================================
Run repeated inference timing experiments and statistical significance analysis
for UNet, UNet-LoMix, and FNO using the latest checkpoints.

This script measures runtime on the fixed test split multiple times per model,
computes omnibus + pairwise significance tests, and generates box plots.

It does not retrain models.

Usage
-----
    python scripts/run_repeated_significance.py --config configs/default.yaml
    python scripts/run_repeated_significance.py --config configs/default.yaml --repeats 30 --discard-first 5
    python scripts/run_repeated_significance.py --config configs/default.yaml --repeats 30 --out-dir results/runtime_significance
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml

# Prevent Intel OpenMP duplicate-runtime crashes on some Windows environments.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    from scipy import stats  # type: ignore[reportMissingImports]
except Exception:  # pragma: no cover
    stats = None

import matplotlib

matplotlib.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 15,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
        "figure.dpi": 180,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.30,
    }
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import build_dataloaders
from src.models.fno import FNO2d
from src.models.unet import UNet2d, UNet2dLoMix


MODEL_ORDER = ["U-Net", "U-Net (LoMix)", "FNO"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Repeated runtime significance for UNet/UNet-LoMix/FNO")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--seed-start", type=int, default=100)
    p.add_argument("--out-dir", default="results/runtime_significance")
    p.add_argument("--warmup", type=int, default=1, help="Untimed warmup passes per model")
    p.add_argument(
        "--discard-first",
        type=int,
        default=0,
        help="Drop the first N repeated timings per model before summary/tests/plots",
    )
    return p.parse_args()


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def append_row(csv_path: Path, row: dict) -> None:
    write_header = (not csv_path.exists()) or csv_path.stat().st_size == 0
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["repeat", "seed", "model", "inference_time_s"])
        writer.writerow([row["repeat"], row["seed"], row["model"], f"{row['inference_time_s']:.10e}"])


def load_config(path: str) -> dict:
    cfg_path = ROOT / path
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)
    cfg["data"]["base_path"] = str((ROOT / "../data").resolve())
    return cfg


def load_state(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _time_unet(model: torch.nn.Module, test_loader, device: torch.device, ini: int) -> float:
    _sync_if_cuda(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        for xx, yy in test_loader:
            xx, yy = xx.to(device), yy.to(device)
            pred = yy[..., :ini, :]
            inp_shp = list(xx.shape[:-2]) + [-1]
            for _ in range(ini, yy.shape[-2]):
                inp = xx.reshape(inp_shp).permute(0, -1, *range(1, len(inp_shp) - 1))
                out = model(inp)
                im = out.permute(0, *range(2, out.ndim), 1).unsqueeze(-2)
                pred = torch.cat([pred, im], dim=-2)
                xx = torch.cat([xx[..., 1:, :], im], dim=-2)
    _sync_if_cuda(device)
    return time.perf_counter() - t0


def _time_fno(model: torch.nn.Module, test_loader_grid, device: torch.device, ini: int) -> float:
    _sync_if_cuda(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        for xx, yy, grid in test_loader_grid:
            xx, yy, grid = xx.to(device), yy.to(device), grid.to(device)
            pred = yy[..., :ini, :]
            inp_shp = list(xx.shape[:-2]) + [-1]
            for _ in range(ini, yy.shape[-2]):
                inp_flat = xx.reshape(inp_shp)
                im = model(inp_flat, grid)
                pred = torch.cat([pred, im], dim=-2)
                xx = torch.cat([xx[..., 1:, :], im], dim=-2)
    _sync_if_cuda(device)
    return time.perf_counter() - t0


def holm_adjust(pvals: List[float]) -> List[float]:
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


def drop_first_rows(rows: List[dict], labels: List[str], drop_first: int) -> List[dict]:
    if drop_first <= 0:
        return rows

    filtered: List[dict] = []
    for label in labels:
        model_rows = sorted([r for r in rows if r["model"] == label], key=lambda r: int(r["repeat"]))
        if drop_first >= len(model_rows):
            raise ValueError(f"discard-first={drop_first} leaves no runs for {label} (available={len(model_rows)})")
        filtered.extend(model_rows[drop_first:])
    return filtered


def make_plot(
    out_path: Path,
    labels: List[str],
    data: List[np.ndarray],
    pairwise_rows: List[dict],
    center: str = "mean",
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[warn] Skipping plot generation: matplotlib import failed: {exc}")
        return False

    fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=180)

    if center not in {"mean", "median"}:
        raise ValueError(f"Unsupported center statistic: {center}")

    means = [float(np.mean(arr)) for arr in data]
    stds = [float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0 for arr in data]
    medians = [float(np.median(arr)) for arr in data]
    centers = means if center == "mean" else medians
    center_label = "mean" if center == "mean" else "median"
    positions = [1.0 + 0.78 * i for i in range(len(labels))]
    rng_boot = np.random.default_rng(42)
    conf_intervals = []
    for arr in data:
        boot_stats = []
        for _ in range(5000):
            sample = rng_boot.choice(arr, size=arr.size, replace=True)
            boot_stats.append(float(np.mean(sample)) if center == "mean" else float(np.median(sample)))
        conf_intervals.append([float(np.percentile(boot_stats, 2.5)), float(np.percentile(boot_stats, 97.5))])

    box_kwargs = dict(
        patch_artist=True,
        notch=True,
        showmeans=False,
        usermedians=centers,
        conf_intervals=np.asarray(conf_intervals, dtype=float),
        positions=positions,
        widths=0.5,
        whis=1.5,
        medianprops={"linewidth": 2.4, "color": "#222"},
        whiskerprops={"linewidth": 1.4, "color": "#555"},
        capprops={"linewidth": 1.4, "color": "#555"},
        boxprops={"linewidth": 1.2, "color": "#444"},
    )
    try:
        box = ax.boxplot(data, tick_labels=labels, **box_kwargs)
    except TypeError:
        box = ax.boxplot(data, labels=labels, **box_kwargs)

    palette = ["#59A14F", "#E15759", "#4C78A8"]
    for patch, color in zip(box["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.30)

    n_runs = int(data[0].size) if data else 0
    if all(int(arr.size) == n_runs for arr in data):
        ax.set_title(f"Inference Runtime Distribution Across {n_runs} Repeated Runs")
    else:
        ax.set_title("Inference Runtime Distribution Across Repeated Runs")
    ax.set_xlabel("Model")
    ax.set_ylabel("Inference time (s)")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.text(
        0.01,
        0.02,
        f"Notches: 95% bootstrap CI of the {center_label}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=13,
        color="#333",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "#f7f7f7", "edgecolor": "#bbb", "alpha": 0.9},
    )

    max_y = max(float(np.max(arr)) for arr in data)
    min_y = min(float(np.min(arr)) for arr in data)
    span = max(1e-12, max_y - min_y)
    left = positions[0] - 0.45
    right = positions[-1] + 0.45
    ax.set_xlim(left, right)

    for pos, cval in zip(positions, centers):
        ax.scatter([pos], [cval], color="#111", s=20, zorder=3)
        ax.text(pos, cval + 0.01 * span, f"{center_label}={cval:.3f}", ha="center", va="bottom", fontsize=11, color="#222")

    if center == "median":
        from matplotlib.lines import Line2D

        legend_handles = []
        for lab, pos, mu, sd, color in zip(labels, positions, means, stds, palette):
            ax.scatter([pos], [mu], marker="x", color=color, s=60, linewidths=2.0, zorder=4)
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="x",
                    color=color,
                    linestyle="None",
                    markersize=8,
                    markeredgewidth=2.0,
                    label=f"{lab}: {mu:.3f} ± {sd:.3f} s",
                )
            )
        ax.legend(handles=legend_handles, loc="upper left", framealpha=0.9, title="Mean ± std")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    out_dir = ROOT / args.out_dir
    ensure_dir(out_dir)
    csv_path = out_dir / "repeated_runs.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ini = int(cfg["data"]["initial_step"])
    discard_first = max(0, min(int(args.discard_first), max(0, int(args.repeats))))

    print(f"[runtime_significance] Device: {device}")

    train_loader, _, test_loader = build_dataloaders(cfg, with_grid=False, max_samples=-1)
    train_loader_grid, _, test_loader_grid = build_dataloaders(cfg, with_grid=True, max_samples=-1)
    sample_xx, _ = next(iter(train_loader))
    sample_xx_grid, _, _ = next(iter(train_loader_grid))
    _, _, _, t, c = sample_xx.shape

    ckpt_dir = ROOT / cfg["paths"]["checkpoint_dir"]
    models = {
        "U-Net": {
            "path": ckpt_dir / "unet_best.pt",
            "loader": test_loader,
            "builder": lambda: UNet2d.from_config(cfg, sample_xx, use_lomix=False),
            "timing": _time_unet,
        },
        "U-Net (LoMix)": {
            "path": ckpt_dir / "unet_lomix_best.pt",
            "loader": test_loader,
            "builder": lambda: UNet2d.from_config(cfg, sample_xx, use_lomix=True),
            "timing": _time_unet,
        },
        "FNO": {
            "path": ckpt_dir / "fno_best.pt",
            "loader": test_loader_grid,
            "builder": lambda: FNO2d.from_config(cfg, sample_xx_grid),
            "timing": _time_fno,
        },
    }

    loaded_models: dict[str, torch.nn.Module] = {}
    for label, info in models.items():
        ckpt_path = info["path"]
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint for {label}: {ckpt_path}")
        loaded_models[label] = load_state(info["builder"](), ckpt_path, device)
        print(f"[runtime_significance] Loaded {label} from {ckpt_path}")

    all_rows: List[dict] = []
    print(f"[runtime_significance] Repeats per model: {args.repeats}")
    for rep in range(args.repeats):
        seed = int(args.seed_start) + rep
        print(f"\n=== Repeat {rep + 1}/{args.repeats} | seed={seed} ===")

        for label in MODEL_ORDER:
            model = loaded_models[label]
            timing_fn = models[label]["timing"]
            loader = models[label]["loader"]

            for _ in range(max(0, int(args.warmup))):
                _ = timing_fn(model, loader, device, ini)

            runtime_s = timing_fn(model, loader, device, ini)
            row = {
                "repeat": rep + 1,
                "seed": seed,
                "model": label,
                "inference_time_s": float(runtime_s),
            }
            all_rows.append(row)
            append_row(csv_path, row)
            print(f"  -> {label}: inference_time_s={runtime_s:.4f}s")

    all_rows_used = drop_first_rows(all_rows, MODEL_ORDER, discard_first)
    if discard_first > 0:
        print(f"[runtime_significance] Dropped first {discard_first} run(s) per model for analysis.")

    by_model: Dict[str, np.ndarray] = {}
    for label in MODEL_ORDER:
        vals = [r["inference_time_s"] for r in all_rows_used if r["model"] == label]
        by_model[label] = np.asarray(vals, dtype=float)

    summary = {
        label: {
            "n": int(arr.size),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
            "median": float(np.median(arr)),
            "q1": float(np.percentile(arr, 25)),
            "q3": float(np.percentile(arr, 75)),
        }
        for label, arr in by_model.items()
    }

    groups = [by_model[label] for label in MODEL_ORDER]
    omnibus = {"test": None, "statistic": None, "p_value": None}
    pairwise_rows: List[dict] = []

    if stats is not None:
        kw = stats.kruskal(*groups)
        omnibus = {"test": "kruskal", "statistic": float(kw.statistic), "p_value": float(kw.pvalue)}

        pairs = list(itertools.combinations(MODEL_ORDER, 2))
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
                }
            )

        adj_ps = holm_adjust(raw_ps)
        for row, p_adj in zip(raw_rows, adj_ps):
            row["p_holm"] = float(p_adj)
            row["significant_0.05"] = bool(p_adj < 0.05)
            pairwise_rows.append(row)
    else:
        print("[warn] scipy is unavailable; significance tests were skipped.")

    plot_path = out_dir / "boxplot_inference_time_s.png"
    plot_ok = make_plot(plot_path, MODEL_ORDER, [by_model[label] for label in MODEL_ORDER], pairwise_rows, center="mean")
    plot_path_median = out_dir / "boxplot_inference_time_s_median.png"
    plot_ok_median = make_plot(plot_path_median, MODEL_ORDER, [by_model[label] for label in MODEL_ORDER], pairwise_rows, center="median")

    summary_payload = {
        "config": args.config,
        "repeats": args.repeats,
        "discard_first": discard_first,
        "seed_start": args.seed_start,
        "metric": "inference_time_s",
        "summary": summary,
        "omnibus": omnibus,
        "pairwise": pairwise_rows,
        "artifacts": {
            "runs_csv": str(csv_path.relative_to(ROOT)),
            "boxplot": str(plot_path.relative_to(ROOT)) if plot_ok else None,
            "boxplot_median": str(plot_path_median.relative_to(ROOT)) if plot_ok_median else None,
        },
    }

    summary_path = out_dir / "runtime_significance_summary.json"
    write_json(summary_path, summary_payload)

    print("\n=== Runtime Significance Results ===")
    print(f"Runs CSV: {csv_path}")
    print(f"Summary JSON: {summary_path}")
    print(f"Box plot: {plot_path}")
    print(f"Box plot (median): {plot_path_median}")
    if pairwise_rows:
        print("\nPairwise tests (Holm-adjusted):")
        for row in pairwise_rows:
            sig = "significant" if row["significant_0.05"] else "ns"
            print(
                f"{row['model_a']} vs {row['model_b']}: p_raw={row['p_raw']:.4g}, "
                f"p_holm={row['p_holm']:.4g} -> {sig}"
            )


if __name__ == "__main__":
    main()
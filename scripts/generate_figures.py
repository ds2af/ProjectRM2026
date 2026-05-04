"""
Generate four publication figures for the three-model SWE surrogate comparison.

AI Disclaimer
-------------
Coding assistance from ChatGPT and GitHub Copilot was used during development.
The author has thoroughly reviewed, checked, and verified the code for correctness
and takes responsibility for the final implementation used in this project.

Figures produced
----------------
1. figures/fig3_field_comparison.png
2. figures/fig4_error_comparison.png  (RMSE per timestep, merged across models)
2b. figures/fig4_error_comparison_bar.png  (bar-chart summary: RMSE, Relative L2, Max Error)
3. figures/fig5_speedup.png
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

# Prevent Intel OpenMP duplicate-runtime crash on some Windows environments
# when torch/numpy stacks load different OpenMP variants in subprocesses.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

matplotlib.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.30,
    }
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import SWEDataset, SWEDatasetWithGrid
from src.models.fno import FNO2d
from src.models.unet import UNet2d, UNet2dLoMix
from src.utils.metrics import compute_all_metrics

MODEL_ORDER = ["unet", "unet_lomix", "fno"]
DEFAULT_MODEL_ORDER = list(MODEL_ORDER)
MODEL_LABELS = {
    "unet": "U-Net",
    "unet_lomix": "U-Net (LoMix)",
    "fno": "FNO",
}
COLORS = {
    "unet": "#59A14F",
    "unet_lomix": "#E15759",
    "fno": "#4C78A8",
}

FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DPI = 150

# Axis typography for Figure 4 and Figure 5 panels.
FIG45_AXIS_LABEL_FONTSIZE = 13
FIG45_TICK_LABEL_FONTSIZE = 11


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument(
        "--exclude_models",
        default="",
        help="Comma-separated model keys to exclude from all generated figures",
    )
    p.add_argument("--t_index", type=int, default=50)
    p.add_argument(
        "--t_indices",
        type=str,
        default="",
        help="Comma-separated timesteps for multi-field comparison (e.g., 10,30,50)",
    )
    p.add_argument(
        "--combine_t_indices",
        action="store_true",
        help="Also save an extra combined Figure 3 copy as fig3_field_comparison_multistep.png",
    )
    p.add_argument(
        "--rmse_steps",
        type=int,
        default=200,
        help="Requested number of timesteps for Figure 4 RMSE curves (clamped to available data)",
    )
    p.add_argument(
        "--rmse_samples",
        type=int,
        default=-1,
        help="Number of test seeds for Figure 4 RMSE curves (-1 uses all test seeds)",
    )
    p.add_argument(
        "--rmse_stride",
        type=int,
        default=1,
        help="Temporal sampling stride for Figure 4 RMSE timeline (1 = every timestep, 2 = every other timestep, etc.)",
    )
    p.add_argument("--dpi", type=int, default=150, help="Output image DPI")
    return p.parse_args()


def save(fig: plt.Figure, name: str):
    out = FIG_DIR / name
    fig.savefig(out, dpi=OUTPUT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] Saved -> {out}")


def ensure_min_timesteps(timesteps: list[int], t_max: int, ini: int, min_count: int = 3) -> list[int]:
    out: list[int] = []
    for t in timesteps:
        t_eff = min(max(0, int(t)), t_max)
        if t_eff not in out:
            out.append(t_eff)

    if len(out) >= min_count:
        return out

    candidates = [ini, t_max, (ini + t_max) // 2, t_max // 3, (2 * t_max) // 3, 0]
    for c in candidates:
        c_eff = min(max(0, int(c)), t_max)
        if c_eff not in out:
            out.append(c_eff)
        if len(out) >= min_count:
            break

    return out


def load_config(path: str) -> dict:
    cfg_path = ROOT / path
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)
    cfg["data"]["base_path"] = str((ROOT / "../data").resolve())
    return cfg


def load_summary() -> dict:
    path = ROOT / "results" / "metrics_summary.json"
    if not path.exists():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "evaluate_all.py")], check=True, cwd=ROOT)
    with path.open() as f:
        return json.load(f)


def is_finite(v) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(v)


def load_state(model: torch.nn.Module, ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        print(f"  [warn] Skipping incompatible checkpoint {ckpt_path.name}: {exc}")
        return None
    model.to(device)
    model.eval()
    return model


def build_models(cfg: dict, sample_xx: torch.Tensor, device: torch.device) -> dict[str, torch.nn.Module]:
    ckpt_dir = ROOT / cfg["paths"]["checkpoint_dir"]
    models: dict[str, torch.nn.Module] = {}

    ckpt = ckpt_dir / "unet_best.pt"
    if ckpt.exists():
        m = load_state(UNet2d.from_config(cfg, sample_xx.unsqueeze(0), use_lomix=False), ckpt, device)
        if m is not None:
            models["unet"] = m

    ckpt = ckpt_dir / "unet_lomix_best.pt"
    if ckpt.exists():
        m = load_state(UNet2d.from_config(cfg, sample_xx.unsqueeze(0), use_lomix=True), ckpt, device)
        if m is not None:
            models["unet_lomix"] = m

    ckpt = ckpt_dir / "fno_best.pt"
    if ckpt.exists():
        m = load_state(FNO2d.from_config(cfg, sample_xx.unsqueeze(0)), ckpt, device)
        if m is not None:
            models["fno"] = m

    missing = [m for m in MODEL_ORDER if m not in models]
    if missing:
        print(f"  [warn] Missing checkpoints for: {', '.join(missing)}")
    return models


def rollout_conv_like(model: torch.nn.Module, xx: torch.Tensor, yy: torch.Tensor, ini: int, device: torch.device) -> torch.Tensor:
    xx_d = xx.unsqueeze(0).to(device)  # [1,H,W,ini,C]
    yy_d = yy.unsqueeze(0).to(device)
    pred = yy_d[..., :ini, :]
    return pred.squeeze(0).cpu()


def rollout_fno(
    model: torch.nn.Module,
    xx: torch.Tensor,
    yy: torch.Tensor,
    grid: torch.Tensor,
    ini: int,
    device: torch.device,
) -> torch.Tensor:
    xx_d = xx.unsqueeze(0).to(device)
    yy_d = yy.unsqueeze(0).to(device)
    grid_d = grid.unsqueeze(0).to(device)
    pred = yy_d[..., :ini, :]

    inp_shape = list(xx_d.shape[:-2]) + [-1]
    for _ in range(ini, yy_d.shape[-2]):
        inp_flat = xx_d.reshape(inp_shape)
        im = model(inp_flat, grid_d)
        pred = torch.cat([pred, im], dim=-2)
        xx_d = torch.cat([xx_d[..., 1:, :], im], dim=-2)

    return pred.squeeze(0).cpu()


def predict_trajectory(
    model_key: str,
    model: torch.nn.Module,
    xx: torch.Tensor,
    yy: torch.Tensor,
    grid: torch.Tensor | None,
    ini: int,
    device: torch.device,
    cfg: dict,
) -> torch.Tensor:
    if model_key == "fno":
        if grid is None:
            raise ValueError("FNO requires grid tensor")
        return rollout_fno(model, xx, yy, grid, ini, device)
    return rollout_conv_like(model, xx, yy, ini, device)


def _style_fig3_panel_frame(ax, color: str) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(color)
        spine.set_linewidth(1.6)


def fig3_field_comparison(
    cfg: dict,
    models: dict[str, torch.nn.Module],
    device: torch.device,
    t_index: int,
    out_name: str = "fig3_field_comparison.png",
):
    ini = cfg["data"]["initial_step"]
    ds = SWEDataset(
        filename=cfg["data"]["filename"],
        saved_folder=cfg["data"]["base_path"],
        initial_step=ini,
        if_test=True,
        test_ratio=cfg["data"].get("test_ratio", 0.1),
        val_ratio=cfg["data"].get("val_ratio", 0.1),
    )
    dsg = SWEDatasetWithGrid(
        filename=cfg["data"]["filename"],
        saved_folder=cfg["data"]["base_path"],
        initial_step=ini,
        if_test=True,
        test_ratio=cfg["data"].get("test_ratio", 0.1),
        val_ratio=cfg["data"].get("val_ratio", 0.1),
    )

    xx, yy = ds[0]
    _, _, grid = dsg[0]
    t_idx = min(max(0, t_index), yy.shape[-2] - 1)

    predictions: dict[str, np.ndarray] = {}
    active = [k for k in MODEL_ORDER if k in models]
    for key in active:
        pred = predict_trajectory(key, models[key], xx, yy, grid if key == "fno" else None, ini, device, cfg)
        predictions[key] = pred[:, :, t_idx, 0].numpy()

    gt = yy[:, :, t_idx, 0].numpy()

    n_cols = 1 + len(active)
    fig, axes = plt.subplots(1, n_cols, figsize=(2.4 * n_cols, 2.8))
    if n_cols == 1:
        axes = [axes]

    fields = [gt] + [predictions[k] for k in active]
    vmin = min(f.min() for f in fields)
    vmax = max(f.max() for f in fields)

    im = axes[0].imshow(gt, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="equal")
    axes[0].set_title("Ground Truth", fontweight="bold", fontsize=11)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    _style_fig3_panel_frame(axes[0], "#333333")

    for i, key in enumerate(active, start=1):
        im = axes[i].imshow(predictions[key], origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="equal")
        axes[i].set_title(MODEL_LABELS[key], fontweight="bold", color=COLORS[key], fontsize=11)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        _style_fig3_panel_frame(axes[i], COLORS[key])

    fig.suptitle(f"Figure 3. Comparison of SWE solutions and ML predictions (timestep {t_idx})", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0.0, 0.0, 0.90, 0.93])
    cax = fig.add_axes([0.915, 0.16, 0.014, 0.68])
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=9)
    save(fig, out_name)


def fig3_field_comparison_combined(
    cfg: dict,
    models: dict[str, torch.nn.Module],
    device: torch.device,
    t_indices: list[int],
    out_name: str = "fig3_field_comparison_multistep.png",
):
    ini = cfg["data"]["initial_step"]
    ds = SWEDataset(
        filename=cfg["data"]["filename"],
        saved_folder=cfg["data"]["base_path"],
        initial_step=ini,
        if_test=True,
        test_ratio=cfg["data"].get("test_ratio", 0.1),
        val_ratio=cfg["data"].get("val_ratio", 0.1),
    )
    dsg = SWEDatasetWithGrid(
        filename=cfg["data"]["filename"],
        saved_folder=cfg["data"]["base_path"],
        initial_step=ini,
        if_test=True,
        test_ratio=cfg["data"].get("test_ratio", 0.1),
        val_ratio=cfg["data"].get("val_ratio", 0.1),
    )

    xx, yy = ds[0]
    _, _, grid = dsg[0]

    active = [k for k in MODEL_ORDER if k in models]
    predictions: dict[str, np.ndarray] = {}
    for key in active:
        pred = predict_trajectory(key, models[key], xx, yy, grid if key == "fno" else None, ini, device, cfg)
        predictions[key] = pred[..., 0].numpy()  # [H, W, T]

    gt_all = yy[..., 0].numpy()  # [H, W, T]
    t_max = gt_all.shape[-1] - 1

    effective_t: list[int] = []
    for t in t_indices:
        t_eff = min(max(0, t), t_max)
        if t_eff != t:
            print(f"  [warn] Requested timestep {t} is out of range [0, {t_max}], using {t_eff}")
        effective_t.append(t_eff)

    fields = []
    for t_eff in effective_t:
        fields.append(gt_all[:, :, t_eff])
        for key in active:
            fields.append(predictions[key][:, :, t_eff])

    vmin = min(f.min() for f in fields)
    vmax = max(f.max() for f in fields)

    n_rows = len(t_indices)
    n_cols = 1 + len(active)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.35 * n_cols, 1.95 * n_rows), squeeze=False)

    image_handle = None
    for r, (t_req, t_eff) in enumerate(zip(t_indices, effective_t)):
        image_handle = axes[r, 0].imshow(gt_all[:, :, t_eff], origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="equal")
        if r == 0:
            axes[r, 0].set_title("Ground Truth", fontweight="bold", fontsize=11)

        row_label = f"t={t_req}" if t_req == t_eff else f"t={t_req} (->{t_eff})"
        axes[r, 0].set_ylabel(row_label, fontweight="bold", fontsize=11)
        axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])
        _style_fig3_panel_frame(axes[r, 0], "#333333")

        for c, key in enumerate(active, start=1):
            image_handle = axes[r, c].imshow(predictions[key][:, :, t_eff], origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="equal")
            if r == 0:
                axes[r, c].set_title(MODEL_LABELS[key], fontweight="bold", color=COLORS[key], fontsize=11)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            _style_fig3_panel_frame(axes[r, c], COLORS[key])

    t_text = ", ".join(str(t) for t in t_indices)
    fig.suptitle(f"Figure 3. SWE field comparison across timesteps [{t_text}]", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 0.90, 0.93])
    if image_handle is not None:
        cax = fig.add_axes([0.915, 0.14, 0.014, 0.72])
        cbar = fig.colorbar(image_handle, cax=cax)
        cbar.ax.tick_params(labelsize=9)
    save(fig, out_name)


def compute_rmse_per_timestep(
    cfg: dict,
    models: dict[str, torch.nn.Module],
    device: torch.device,
    max_steps: int = 200,
    max_samples: int = -1,
    step_stride: int = 1,
) -> dict:
    ini = cfg["data"]["initial_step"]
    ds = SWEDataset(
        filename=cfg["data"]["filename"],
        saved_folder=cfg["data"]["base_path"],
        initial_step=ini,
        if_test=True,
        test_ratio=cfg["data"].get("test_ratio", 0.1),
        val_ratio=cfg["data"].get("val_ratio", 0.1),
    )
    dsg = SWEDatasetWithGrid(
        filename=cfg["data"]["filename"],
        saved_folder=cfg["data"]["base_path"],
        initial_step=ini,
        if_test=True,
        test_ratio=cfg["data"].get("test_ratio", 0.1),
        val_ratio=cfg["data"].get("val_ratio", 0.1),
    )

    n_total = len(ds)
    if n_total == 0:
        return {}

    n_eval = n_total if max_samples < 0 else min(max_samples, n_total)
    _, yy0 = ds[0]
    t_avail = int(yy0.shape[-2])
    req_steps = max(1, int(max_steps))
    n_steps = min(req_steps, t_avail)
    stride = max(1, int(step_stride))
    if req_steps > t_avail:
        print(f"  [warn] Requested {req_steps} RMSE timesteps, but only {t_avail} available; using {n_steps}")

    t_indices = np.arange(0, n_steps, stride, dtype=int)
    if t_indices.size == 0:
        t_indices = np.array([0], dtype=int)
    if stride > 1:
        print(f"  [fig4] Using timestep stride={stride} -> {t_indices.size} sampled points")

    active = [k for k in MODEL_ORDER if k in models]
    if not active:
        return {}

    need_grid = "fno" in active
    mse_sum = {k: np.zeros(t_indices.size, dtype=np.float64) for k in active}

    for i in range(n_eval):
        xx, yy = ds[i]
        grid = None
        if need_grid:
            _, _, grid = dsg[i]

        yy_slice = yy[:, :, t_indices, :]
        for key in active:
            pred = predict_trajectory(key, models[key], xx, yy, grid if key == "fno" else None, ini, device, cfg)
            pred_slice = pred[:, :, t_indices, :]
            mse_t = torch.mean((pred_slice - yy_slice) ** 2, dim=(0, 1, 3))
            mse_sum[key] += mse_t.detach().cpu().numpy()

    rmse_curves = {k: np.sqrt(mse_sum[k] / max(1, n_eval)) for k in active}
    payload = {
        "timesteps": [int(t) for t in t_indices.tolist()],
        "n_steps": int(t_indices.size),
        "requested_steps": req_steps,
        "step_stride": stride,
        "n_eval_seeds": n_eval,
        "initial_step": ini,
        "rmse": {k: [float(v) for v in rmse_curves[k]] for k in active},
    }

    out_path = ROOT / "results" / "rmse_per_timestep.json"
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"  [fig4] Saved RMSE timeline data -> {out_path}")
    return payload


def fig4_error_comparison(
    cfg: dict,
    models: dict[str, torch.nn.Module],
    device: torch.device,
    max_steps: int = 200,
    max_samples: int = -1,
    step_stride: int = 1,
):
    rmse_data = compute_rmse_per_timestep(
        cfg,
        models,
        device,
        max_steps=max_steps,
        max_samples=max_samples,
        step_stride=step_stride,
    )
    if not rmse_data:
        print("  [warn] Skipping Figure 4: RMSE per-timestep data not available")
        return

    t = np.array(rmse_data["timesteps"], dtype=int)
    ini = int(rmse_data["initial_step"])
    active = [k for k in MODEL_ORDER if k in rmse_data["rmse"]]
    if not active:
        print("  [warn] Skipping Figure 4: no RMSE curves available")
        return

    t_train = int(cfg["data"].get("t_train", 40))
    t_max = int(t.max())

    fig, ax = plt.subplots(1, 1, figsize=(10.5, 4.6))

    # ── Shaded regions ──────────────────────────────────────────────────
    ax.axvspan(int(t.min()), ini, alpha=0.10, color="#888888",
               label=f"Context (GT input, t<{ini})")
    ax.axvspan(ini, min(t_train, t_max), alpha=0.10, color="#4C78A8",
               label=f"AR training window (t={ini}–{t_train}, U-Net/FNO supervised)")
    if t_train < t_max:
        ax.axvspan(t_train, t_max, alpha=0.07, color="#E45756",
                   label=f"Extrapolation (t>{t_train})")

    for key in active:
        y = np.array(rmse_data["rmse"][key], dtype=float)
        ax.plot(t, y, color=COLORS[key], linewidth=2.0, linestyle="-", label=MODEL_LABELS[key])

    ax.axvline(ini, color="#444444", linestyle=":", linewidth=1.0)
    ax.axvline(t_train, color="#444444", linestyle=":", linewidth=1.0)

    ax.set_xlabel("Timestep", fontsize=FIG45_AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("RMSE", fontsize=FIG45_AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=FIG45_TICK_LABEL_FONTSIZE)
    ax.set_title(
        f"RMSE per timestep - AR training window t={ini}-{t_train}",
        fontweight="bold", fontsize=10,
    )
    ax.set_xlim(int(t.min()), t_max)
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0, ncol=1, fontsize=8)

    step_stride = int(rmse_data.get("step_stride", 1))
    stride_note = f", stride={step_stride}" if step_stride > 1 else ""
    fig.suptitle(
        f"Figure 4. RMSE timeline across models ({rmse_data['n_steps']} steps{stride_note}, {rmse_data['n_eval_seeds']} test seeds)",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0.0, 0.0, 0.76, 0.95])
    save(fig, "fig4_error_comparison.png")


def fig4_error_comparison_bar(summary: dict):
    models = [m for m in MODEL_ORDER if m in summary]
    if not models:
        print("  [warn] Skipping Figure 4 bar chart: summary metrics not available")
        return

    labels = [MODEL_LABELS[m] for m in models]
    colors = [COLORS[m] for m in models]
    rmse = [summary[m].get("rmse", float("nan")) for m in models]
    rel_l2 = [summary[m].get("rel_l2", float("nan")) for m in models]

    x = np.arange(len(models))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))

    panels = [
        (axes[0], rmse, "RMSE"),
        (axes[1], rel_l2, "Relative L2"),
    ]

    for ax, values, ylabel in panels:
        bars = ax.bar(x, values, color=colors, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=FIG45_TICK_LABEL_FONTSIZE)
        ax.set_ylabel(ylabel, fontsize=FIG45_AXIS_LABEL_FONTSIZE)
        ax.tick_params(axis="y", labelsize=FIG45_TICK_LABEL_FONTSIZE)
        ax.set_title(ylabel, fontweight="bold")

        finite_vals = [float(v) for v in values if is_finite(v)]
        if finite_vals:
            ymax = max(finite_vals)
            ax.set_ylim(0.0, ymax * 1.18 if ymax > 0 else 1.0)

        for i, (b, v) in enumerate(zip(bars, values)):
            if is_finite(v):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    b.get_height() * 1.02,
                    f"{float(v):.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="black",
                )

    fig.suptitle("Figure 4b. Error comparison bar chart across models", fontsize=11, fontweight="bold")
    fig.text(
        0.5,
        0.01,
        "",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=[0.0, 0.04, 1.0, 0.95])
    save(fig, "fig4_error_comparison_bar.png")


def fig5_speedup(summary: dict):
    models = [m for m in MODEL_ORDER if m in summary]
    if not models:
        print("  [warn] Skipping Figure 5: summary metrics not available")
        return

    labels = [MODEL_LABELS[m] for m in models]
    colors = [COLORS[m] for m in models]
    inf_t = [summary[m].get("inference_time_s", float("nan")) for m in models]

    x = np.arange(len(models))
    fig, ax = plt.subplots(1, 1, figsize=(5, 4.2))

    bars = ax.bar(x, inf_t, color=colors, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=FIG45_TICK_LABEL_FONTSIZE)
    ax.set_ylabel("Inference time (s)", fontsize=FIG45_AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="y", labelsize=FIG45_TICK_LABEL_FONTSIZE)
    ax.set_title("Inference Time (s)", fontweight="bold")

    finite_vals = [float(v) for v in inf_t if is_finite(v)]
    if finite_vals:
        ymax = max(finite_vals)
        ax.set_ylim(0.0, ymax * 1.18 if ymax > 0 else 1.0)

    for i, (b, v) in enumerate(zip(bars, inf_t)):
        if is_finite(v):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() * 1.02,
                f"{float(v):.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="black",
            )

    fig.suptitle("Figure 5. Inference time of ML models", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0.0, 0.02, 1.0, 0.95])
    save(fig, "fig5_speedup.png")


def main():
    global OUTPUT_DPI, MODEL_ORDER
    args = parse_args()
    OUTPUT_DPI = max(72, int(args.dpi))
    matplotlib.rcParams["figure.dpi"] = OUTPUT_DPI
    cfg = load_config(args.config)
    summary = load_summary()

    excluded = {item.strip() for item in args.exclude_models.split(",") if item.strip()}
    MODEL_ORDER = [model for model in DEFAULT_MODEL_ORDER if model not in excluded]
    if excluded:
        print(f"[generate_figures] Excluding models: {', '.join(sorted(excluded))}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[generate_figures] Device: {device}")
    print(f"[generate_figures] Output DPI: {OUTPUT_DPI}")

    ini = cfg["data"]["initial_step"]
    base_ds = SWEDataset(
        filename=cfg["data"]["filename"],
        saved_folder=cfg["data"]["base_path"],
        initial_step=ini,
        if_test=True,
        test_ratio=cfg["data"].get("test_ratio", 0.1),
        val_ratio=cfg["data"].get("val_ratio", 0.1),
    )
    sample_xx, sample_yy = base_ds[0]

    print("[generate_figures] Loading checkpoints ...")
    models = build_models(cfg, sample_xx, device)

    print("[generate_figures] Creating figures ...")

    if args.t_indices.strip():
        requested = [s.strip() for s in args.t_indices.split(",") if s.strip()]
        raw_timesteps: list[int] = []
        for item in requested:
            try:
                raw_timesteps.append(int(item))
            except ValueError:
                print(f"  [warn] Skipping invalid timestep entry: {item}")
        if not raw_timesteps:
            print("  [warn] No valid timesteps in --t_indices; using --t_index.")
            raw_timesteps = [10, 50, 100]
    else:
        raw_timesteps = [10, 50, 100]

    t_max = int(sample_yy.shape[-2] - 1)
    timesteps = ensure_min_timesteps(raw_timesteps, t_max=t_max, ini=ini, min_count=3)
    if len(timesteps) < 3:
        print(f"  [warn] Could not form 3 unique timesteps, using: {timesteps}")
    else:
        print(f"  [fig3] Using timesteps: {timesteps}")

    fig3_field_comparison_combined(cfg, models, device, timesteps, out_name="fig3_field_comparison.png")
    if args.combine_t_indices:
        fig3_field_comparison_combined(cfg, models, device, timesteps, out_name="fig3_field_comparison_multistep.png")

    fig4_error_comparison(
        cfg,
        models,
        device,
        max_steps=args.rmse_steps,
        max_samples=args.rmse_samples,
        step_stride=args.rmse_stride,
    )
    fig4_error_comparison_bar(summary)
    fig5_speedup(summary)

    print(f"[generate_figures] Done. Figures saved in {FIG_DIR}")


if __name__ == "__main__":
    main()

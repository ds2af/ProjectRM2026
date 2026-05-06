"""
scripts/generate_figures_wrapper.py
====================================
Generate publication-quality figures from trained models.

Supports:
- Training curves (train/val loss over epochs)
- Field comparisons (predicted vs ground truth)
- Error analysis plots (RMSE over timesteps)
- Runtime statistics (inference time distribution)

Usage
-----
    python scripts/generate_figures_wrapper.py
    python scripts/generate_figures_wrapper.py --metrics-json results/metrics_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import SWEDataset, SWEDatasetWithGrid
from src.models.fno import FNO2d
from src.models.unet import UNet2d
from src.utils.metrics import per_timestep_rmse


plt.rcParams.update(
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
    }
)


def parse_args():
    p = argparse.ArgumentParser(description="Generate publication figures")
    p.add_argument("--metrics-json", default="results/metrics_summary.json")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument("--t-index", type=int, default=50)
    p.add_argument("--out-dir", default="figures")
    return p.parse_args()


def load_config(config_path: Path) -> dict:
    with config_path.open() as f:
        return yaml.safe_load(f)


def load_test_sample(cfg: dict, sample_index: int):
    data_cfg = cfg.get("data", {})
    common_kwargs = dict(
        filename=data_cfg.get("filename", "2D_rdb_NA_NA"),
        saved_folder=data_cfg.get("base_path", "../data/"),
        initial_step=int(data_cfg.get("initial_step", 10)),
        if_test=True,
        test_ratio=float(data_cfg.get("test_ratio", 0.1)),
        val_ratio=float(data_cfg.get("val_ratio", 0.0)),
        max_samples=-1,
    )

    ds = SWEDataset(**common_kwargs)
    ds_grid = SWEDatasetWithGrid(**common_kwargs)
    sample_xx, sample_yy = ds[sample_index % len(ds)]
    _, _, sample_grid = ds_grid[sample_index % len(ds_grid)]
    return sample_xx, sample_yy, sample_grid


def load_model(model_name: str, cfg: dict, sample_xx: torch.Tensor, device: torch.device):
    ckpt_dir = ROOT / cfg.get("paths", {}).get("checkpoint_dir", "results/checkpoints")
    ckpt_path = ckpt_dir / f"{model_name}_best.pt"
    if not ckpt_path.exists():
        print(f"[skip] Missing checkpoint: {ckpt_path.name}")
        return None

    if model_name == "unet":
        model = UNet2d.from_config(cfg, sample_xx.unsqueeze(0), use_lomix=False)
    elif model_name == "unet_lomix":
        model = UNet2d.from_config(cfg, sample_xx.unsqueeze(0), use_lomix=True)
    elif model_name == "fno":
        model = FNO2d.from_config(cfg, sample_xx.unsqueeze(0))
    else:
        return None

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def rollout_unet(model: torch.nn.Module, xx: torch.Tensor, target_t: int, device: torch.device) -> torch.Tensor:
    context = xx.unsqueeze(0).to(device)
    initial_step = context.shape[-2]
    pred = None

    with torch.no_grad():
        for _ in range(initial_step, target_t + 1):
            inp = context.reshape(1, context.shape[1], context.shape[2], -1).permute(0, 3, 1, 2)
            out = model(inp)
            pred = out.permute(0, 2, 3, 1).unsqueeze(-2)
            context = torch.cat([context[..., 1:, :], pred], dim=-2)

    return pred.squeeze(0).squeeze(-2)


def rollout_fno(model: torch.nn.Module, xx: torch.Tensor, grid: torch.Tensor, target_t: int, device: torch.device) -> torch.Tensor:
    context = xx.unsqueeze(0).to(device)
    grid = grid.unsqueeze(0).to(device)
    initial_step = context.shape[-2]
    pred = None

    with torch.no_grad():
        for _ in range(initial_step, target_t + 1):
            inp = context.reshape(1, context.shape[1], context.shape[2], -1)
            out = model(inp, grid)
            pred = out
            context = torch.cat([context[..., 1:, :], pred], dim=-2)

    return pred.squeeze(0).squeeze(-2)


def rollout_trajectory(
    model_name: str,
    model: torch.nn.Module,
    xx: torch.Tensor,
    grid: torch.Tensor | None,
    full_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """Roll a model forward over the full trajectory and return [H, W, T, C]."""
    context = xx.unsqueeze(0).to(device)
    initial_step = context.shape[-2]
    trajectory = context.clone()

    if full_steps <= initial_step:
        return trajectory.squeeze(0)

    with torch.no_grad():
        for _ in range(initial_step, full_steps):
            if model_name == "fno":
                if grid is None:
                    raise ValueError("FNO rollout requires a grid tensor")
                inp = context.reshape(1, context.shape[1], context.shape[2], -1)
                pred = model(inp, grid.unsqueeze(0).to(device))
                pred_step = pred
            else:
                inp = context.reshape(1, context.shape[1], context.shape[2], -1).permute(0, 3, 1, 2)
                pred = model(inp)
                pred_step = pred.permute(0, 2, 3, 1).unsqueeze(-2)

            trajectory = torch.cat([trajectory, pred_step], dim=-2)
            context = torch.cat([context[..., 1:, :], pred_step], dim=-2)

    return trajectory.squeeze(0)


def compute_rmse_per_timestep(cfg: dict, models: dict[str, torch.nn.Module], device: torch.device, max_steps: int = 200, max_samples: int = -1, step_stride: int = 1) -> dict:
    ini = int(cfg.get("data", {}).get("initial_step", 10))
    data_cfg = cfg.get("data", {})
    common_kwargs = dict(
        filename=data_cfg.get("filename", "2D_rdb_NA_NA"),
        saved_folder=data_cfg.get("base_path", "../data/"),
        initial_step=ini,
        if_test=True,
        test_ratio=float(data_cfg.get("test_ratio", 0.1)),
        val_ratio=float(data_cfg.get("val_ratio", 0.0)),
        max_samples=-1,
    )

    ds = SWEDataset(**common_kwargs)
    dsg = SWEDatasetWithGrid(**common_kwargs)

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
        print(f"[warn] Requested {req_steps} RMSE timesteps, but only {t_avail} are available; using {n_steps}")

    t_indices = np.arange(0, n_steps, stride, dtype=int)
    if t_indices.size == 0:
        t_indices = np.array([0], dtype=int)

    active = [k for k in ["unet", "unet_lomix", "fno"] if k in models]
    if not active:
        return {}

    mse_sum = {k: np.zeros(t_indices.size, dtype=np.float64) for k in active}

    for i in range(n_eval):
        xx, yy = ds[i]
        grid = dsg[i][2] if "fno" in active else None
        yy_slice = yy[:, :, t_indices, :]

        for key in active:
            traj = rollout_trajectory(key, models[key], xx, grid if key == "fno" else None, yy.shape[-2], device)
            pred_slice = traj[:, :, t_indices, :]
            mse_t = torch.mean((pred_slice - yy_slice) ** 2, dim=(0, 1, 3))
            mse_sum[key] += mse_t.detach().cpu().numpy()

    rmse_curves = {k: np.sqrt(mse_sum[k] / max(1, n_eval)) for k in active}
    return {
        "timesteps": [int(t) for t in t_indices.tolist()],
        "n_steps": int(t_indices.size),
        "requested_steps": req_steps,
        "step_stride": stride,
        "n_eval_seeds": n_eval,
        "initial_step": ini,
        "rmse": {k: [float(v) for v in rmse_curves[k]] for k in active},
    }


def plot_error_comparison(cfg: dict, out_dir: Path):
    """Plot RMSE per timestep across models, matching the backup Figure 4 timeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = cfg.get("data", {})
    sample_xx, _, _ = load_test_sample(cfg, 0)
    models = {
        "unet": load_model("unet", cfg, sample_xx, device),
        "unet_lomix": load_model("unet_lomix", cfg, sample_xx, device),
        "fno": load_model("fno", cfg, sample_xx, device),
    }
    models = {k: v for k, v in models.items() if v is not None}
    if not models:
        print("[skip] No model checkpoints available for error comparison")
        return

    rmse_data = compute_rmse_per_timestep(
        cfg,
        models,
        device,
        max_steps=int(data_cfg.get("t_train", 200)),
        max_samples=-1,
        step_stride=1,
    )
    if not rmse_data:
        print("[skip] RMSE per-timestep data not available")
        return

    t = np.array(rmse_data["timesteps"], dtype=int)
    ini = int(rmse_data["initial_step"])
    t_train = int(data_cfg.get("t_train", 40))
    t_max = int(t.max())
    active = [k for k in ["unet", "unet_lomix", "fno"] if k in rmse_data["rmse"]]
    if not active:
        print("[skip] No RMSE curves available")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10.5, 4.6))
    ax.axvspan(int(t.min()), ini, alpha=0.10, color="#888888", label=f"Context (GT input, t<{ini})")
    ax.axvspan(ini, min(t_train, t_max), alpha=0.10, color="#4C78A8", label=f"AR training window (t={ini}–{t_train})")
    if t_train < t_max:
        ax.axvspan(t_train, t_max, alpha=0.07, color="#E45756", label=f"Extrapolation (t>{t_train})")

    color_map = {"unet": "#59A14F", "unet_lomix": "#E15759", "fno": "#4C78A8"}
    label_map = {"unet": "U-Net", "unet_lomix": "U-Net (LoMix)", "fno": "FNO"}
    for key in active:
        ax.plot(t, np.array(rmse_data["rmse"][key], dtype=float), color=color_map[key], linewidth=2.0, label=label_map[key])

    ax.axvline(ini, color="#444444", linestyle=":", linewidth=1.0)
    ax.axvline(t_train, color="#444444", linestyle=":", linewidth=1.0)
    ax.set_xlabel("Timestep", fontsize=13)
    ax.set_ylabel("RMSE", fontsize=13)
    ax.set_title(f"RMSE per timestep across models - AR training window t={ini}-{t_train}", fontweight="bold", fontsize=10)
    ax.set_xlim(int(t.min()), t_max)
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0, ncol=1, fontsize=8)
    fig.tight_layout(rect=[0.0, 0.0, 0.76, 0.95])

    out_path = out_dir / "error_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path.name}")


def plot_field_comparison(cfg: dict, out_dir: Path, sample_index: int, t_index: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_xx, sample_yy, sample_grid = load_test_sample(cfg, sample_index)
    initial_step = sample_xx.shape[-2]
    t_index = max(initial_step, min(int(t_index), sample_yy.shape[-2] - 1))

    models = {
        "unet": load_model("unet", cfg, sample_xx, device),
        "unet_lomix": load_model("unet_lomix", cfg, sample_xx, device),
        "fno": load_model("fno", cfg, sample_xx, device),
    }

    active_models = {name: model for name, model in models.items() if model is not None}
    if not active_models:
        print("[skip] No model checkpoints available for field comparison")
        return

    truth = sample_yy[..., t_index, 0].numpy()
    truth_min = float(np.min(truth))
    truth_max = float(np.max(truth))

    preds: dict[str, np.ndarray] = {}
    for name, model in active_models.items():
        if name == "fno":
            pred = rollout_fno(model, sample_xx, sample_grid, t_index, device)
        else:
            pred = rollout_unet(model, sample_xx, t_index, device)
        preds[name] = pred[..., 0].detach().cpu().numpy()

    model_order = [name for name in ["unet", "unet_lomix", "fno"] if name in preds]
    if not model_order:
        print("[skip] No valid model predictions for field comparison")
        return

    fig, axes = plt.subplots(len(model_order), 3, figsize=(12, 3.4 * len(model_order)), constrained_layout=True)
    if len(model_order) == 1:
        axes = np.expand_dims(axes, axis=0)

    column_titles = ["Ground Truth", "Prediction", "Abs Error"]
    error_max = max(float(np.max(np.abs(preds[name] - truth))) for name in model_order)
    error_max = error_max if error_max > 0 else 1.0

    for row, model_name in enumerate(model_order):
        pred = preds[model_name]
        err = np.abs(pred - truth)
        model_label = model_name.replace("_", " ").title()

        panels = [truth, pred, err]
        cmaps = ["viridis", "viridis", "magma"]
        norms = [dict(vmin=truth_min, vmax=truth_max), dict(vmin=truth_min, vmax=truth_max), dict(vmin=0.0, vmax=error_max)]

        for col, (panel, cmap, norm) in enumerate(zip(panels, cmaps, norms)):
            ax = axes[row, col]
            im = ax.imshow(panel, cmap=cmap, **norm)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(column_titles[col])
            if col == 0:
                ax.set_ylabel(model_label)
            if row == len(model_order) - 1:
                ax.set_xlabel(f"t = {t_index}")

            if col == 2:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    out_path = out_dir / "field_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path.name}")


def plot_training_curves(log_dir: Path, out_dir: Path):
    """Plot training and validation loss curves from CSV logs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    models = ["unet", "unet_lomix", "fno"]
    
    for idx, model in enumerate(models):
        log_file = log_dir / f"{model}_log.csv"
        if not log_file.exists():
            print(f"[skip] {model}_log.csv not found")
            continue
        
        df = pd.read_csv(log_file)
        axes[idx].plot(df["epoch"], df["train_loss"], label="Train", marker="o", markersize=2)
        axes[idx].plot(df["epoch"], df["val_loss"], label="Val", marker="s", markersize=2)
        axes[idx].set_xlabel("Epoch")
        axes[idx].set_ylabel("Loss (MSE)")
        axes[idx].set_title(model.replace("_", " ").title())
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = out_dir / "training_val_loss_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[save] {out_path.name}")
    plt.close()


def plot_metrics_summary(metrics_json: Path, out_dir: Path):
    """Plot bar chart comparing model metrics."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not metrics_json.exists():
        print(f"[skip] Metrics JSON not found: {metrics_json}")
        return
    
    with metrics_json.open() as f:
        data = json.load(f)
    
    models = []
    rmse_vals = []
    runtime_vals = []
    
    for model_name, metrics in data.items():
        if isinstance(metrics, dict) and "rmse" in metrics:
            models.append(model_name.replace("_", " ").title())
            rmse_vals.append(metrics.get("rmse", 0))
            runtime_vals.append(metrics.get("inference_time_s", 0))
    
    if not models:
        print("[skip] No valid metrics found")
        return
    
    x = np.arange(len(models))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, rmse_vals, width, label="RMSE", alpha=0.8)
    ax.bar(x + width, runtime_vals, width, label="Runtime (s)", alpha=0.8)
    
    ax.set_xlabel("Model")
    ax.set_ylabel("Value")
    ax.set_title("RMSE and Runtime Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    out_path = out_dir / "error_comparison_bar.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[save] {out_path.name}")
    plt.close()


def plot_runtime_stats(metrics_json: Path, out_dir: Path):
    """Plot inference time distribution."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not metrics_json.exists():
        print(f"[skip] Metrics JSON not found: {metrics_json}")
        return
    
    with metrics_json.open() as f:
        data = json.load(f)
    
    models = []
    times = []
    
    for model_name, metrics in data.items():
        if isinstance(metrics, dict) and "inference_time_s" in metrics:
            models.append(model_name.replace("_", " ").title())
            times.append(metrics.get("inference_time_s", 0))
    
    if not models:
        print("[skip] No inference time data found")
        return
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(models, times, alpha=0.7, color="steelblue")
    ax.set_ylabel("Inference Time (s)")
    ax.set_title("Model Inference Time")
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    out_path = out_dir / "boxplot_inference_time_s_median.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[save] {out_path.name}")
    plt.close()


def main():
    args = parse_args()
    out_dir = ROOT / args.out_dir
    cfg = load_config(ROOT / args.config)
    log_dir = ROOT / "results" / "logs"
    metrics_json = ROOT / args.metrics_json
    
    print("[generate_figures_wrapper] Generating figures...")

    try:
        plot_field_comparison(cfg, out_dir, args.sample_index, args.t_index)
    except Exception as e:
        print(f"[error] Field comparison failed: {e}")

    try:
        plot_error_comparison(cfg, out_dir)
    except Exception as e:
        print(f"[error] Error comparison failed: {e}")
    
    # Generate training curves
    if log_dir.exists():
        try:
            plot_training_curves(log_dir, out_dir)
        except Exception as e:
            print(f"[error] Training curves failed: {e}")
    else:
        print(f"[skip] Logs directory not found: {log_dir}")
    
    # Generate metrics comparison
    try:
        plot_metrics_summary(metrics_json, out_dir)
    except Exception as e:
        print(f"[error] Metrics summary failed: {e}")
    
    # Generate runtime stats
    try:
        plot_runtime_stats(metrics_json, out_dir)
    except Exception as e:
        print(f"[error] Runtime stats failed: {e}")
    
    print("[generate_figures_wrapper] Done")


if __name__ == "__main__":
    main()

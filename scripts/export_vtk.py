"""
scripts/export_vtk.py
=====================
Export ground truth and model predictions to VTK format for visualization
in ParaView, VisIt, or other scientific visualization tools.

AI Disclaimer
-------------
Coding assistance from ChatGPT and GitHub Copilot was used during development.
The author has thoroughly reviewed, checked, and verified the code for correctness
and takes responsibility for the final implementation used in this project.

Usage
-----
    # Export U-Net predictions
    python scripts/export_vtk.py --model unet_lomix --config configs/default.yaml
    
    # Export all models
    python scripts/export_vtk.py --model all --config configs/default.yaml
    
    # Specify number of trajectories and timesteps
    python scripts/export_vtk.py --model fno --n_samples 2 --n_timesteps 50

VTK files saved to: results/vtk/<model>/
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import SWEDataset, SWEDatasetWithGrid
from src.models.unet import UNet2d, UNet2dLoMix
from src.models.fno import FNO2d


def _resolve_h5_path(cfg: dict) -> Path:
    """Resolve the configured raw HDF5 file path from data folder."""
    p = Path(cfg["data"]["base_path"]) / f"{cfg['data']['filename']}.h5"
    p = p.resolve()
    if p.exists():
        return p
    fallback = (ROOT / "../data" / f"{cfg['data']['filename']}.h5").resolve()
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Raw HDF5 not found: {p} (fallback: {fallback})")


def write_vtk_structured(
    field: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    filename: Path,
    var_name: str = "h"
) -> None:
    """
    Write 2D scalar field to VTK STRUCTURED_POINTS format.
    
    Parameters
    ----------
    field : np.ndarray
        2D field [Ny, Nx] (y-rows, x-columns)
    x_min, x_max, y_min, y_max : float
        Physical domain bounds
    filename : Path
        Output VTK file path
    var_name : str
        Variable name in VTK file
    """
    field = np.asarray(field)
    Ny, Nx = field.shape
    Nz = 1
    
    dx = (x_max - x_min) / max(Nx - 1, 1)
    dy = (y_max - y_min) / max(Ny - 1, 1)
    dz = 1.0
    
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    
    with filename.open("w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"{var_name} field\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {Nx} {Ny} {Nz}\n")
        f.write(f"ORIGIN {x_min} {y_min} 0.0\n")
        f.write(f"SPACING {dx} {dy} {dz}\n")
        f.write(f"POINT_DATA {Nx * Ny * Nz}\n")
        f.write(f"SCALARS {var_name} float 1\n")
        f.write("LOOKUP_TABLE default\n")
        
        for j in range(Ny):
            for i in range(Nx):
                f.write(f"{float(field[j, i])}\n")


def export_model_vtk(
    model_name: str,
    cfg: dict,
    n_samples: int = 1,
    n_timesteps: int = 101,
    x_min: float = 0.0,
    x_max: float = 1.0,
    y_min: float = 0.0,
    y_max: float = 1.0,
) -> None:
    """
    Export VTK files for a specific trained model.
    
    Parameters
    ----------
    model_name : str
        One of: unet, unet_lomix, fno
    cfg : dict
        Configuration dictionary
    n_samples : int
        Number of test trajectories to export
    n_timesteps : int
        Maximum timesteps to export per trajectory
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[export_vtk] Model: {model_name}  Device: {device}")

    # Resolve one canonical raw HDF5 path and reuse it everywhere.
    h5_path = _resolve_h5_path(cfg)
    data_folder = str(h5_path.parent)
    print(f"[export_vtk] Data source: {h5_path}")
    
    # Load checkpoint
    checkpoint_path = ROOT / cfg["paths"]["checkpoint_dir"] / f"{model_name}_best.pt"
    if not checkpoint_path.exists():
        print(f"[export_vtk] Checkpoint not found: {checkpoint_path}")
        print(f"[export_vtk] Skipping {model_name}")
        return
    
    # Build test dataset/loader (batch_size=1 keeps one seed per export sample)
    needs_grid = (model_name == "fno")
    if needs_grid:
        test_dataset = SWEDatasetWithGrid(
            filename=cfg["data"]["filename"],
            saved_folder=data_folder,
            initial_step=cfg["data"]["initial_step"],
            if_test=True,
            test_ratio=cfg["data"]["test_ratio"],
            val_ratio=cfg["data"]["val_ratio"],
        )
    else:
        test_dataset = SWEDataset(
            filename=cfg["data"]["filename"],
            saved_folder=data_folder,
            initial_step=cfg["data"]["initial_step"],
            if_test=True,
            test_ratio=cfg["data"]["test_ratio"],
            val_ratio=cfg["data"]["val_ratio"],
        )

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # Load model
    sample_batch = next(iter(test_loader))
    sample_xx = sample_batch[0]
    B, H, W, T, C = sample_xx.shape
    
    ucfg = cfg.get("unet", {})
    fcfg = cfg.get("fno", {})

    if model_name == "unet":
        model = UNet2d(
            in_channels=C * T,
            out_channels=C,
            init_features=ucfg.get("init_features", 32),
            dropout=ucfg.get("dropout", 0.0)
        ).to(device)
    elif model_name == "unet_lomix":
        model = UNet2dLoMix(
            in_channels=C * T,
            out_channels=C,
            init_features=ucfg.get("init_features", 32),
            dropout=ucfg.get("dropout", 0.0)
        ).to(device)
    elif model_name == "fno":
        model = FNO2d(
            modes1=fcfg.get("modes1", 12),
            modes2=fcfg.get("modes2", 12),
            width=fcfg.get("width", 32),
            num_channels=C,
            initial_step=T
        ).to(device)
    else:
        print(f"[export_vtk] Unknown model: {model_name}")
        return
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[export_vtk] Loaded checkpoint from {checkpoint_path}")
    
    # Export directory
    vtk_dir = ROOT / "results" / "vtk" / model_name
    vtk_dir.mkdir(parents=True, exist_ok=True)
    
    # Run inference and export
    initial_step = cfg["data"]["initial_step"]
    sample_count = 0
    
    total_files = 0
    with h5py.File(h5_path, "r") as hf:
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if sample_count >= n_samples:
                    break

                has_grid = len(batch) == 3
                xx = batch[0].to(device)
                yy = batch[1].to(device)
                grid = batch[2].to(device) if has_grid else None

                # Autoregressive rollout for all models.
                pred = yy[..., :initial_step, :]
                inp_shape = list(xx.shape[:-2]) + [-1]  # [B, H, W, T*C]

                for t in range(initial_step, min(yy.shape[-2], n_timesteps)):
                    inp = xx.reshape(inp_shape)  # [B, H, W, T*C]

                    if model_name in ("unet", "unet_lomix"):
                        inp_chfirst = inp.permute(0, -1, *range(1, inp.ndim - 1))  # [B, T*C, H, W]
                        out = model(inp_chfirst)  # [B, C, H, W]
                        im = out.permute(0, *range(2, out.ndim), 1).unsqueeze(-2)    # [B, H, W, 1, C]

                    elif model_name == "fno":
                        im = model(inp, grid)  # [B, H, W, 1, C]

                    else:
                        raise ValueError(f"Unknown model_name for AR rollout: {model_name}")

                    pred = torch.cat([pred, im], dim=-2)
                    xx = torch.cat([xx[..., 1:, :], im], dim=-2)  # slide context window forward

                pred_np = pred[0].cpu().numpy()  # [H, W, T_pred, C]

                # Ground truth is read directly from raw HDF5 in data folder.
                seed_key = str(test_dataset.data_list[batch_idx])
                raw_truth = np.array(hf[seed_key]["data"], dtype=np.float32)  # [T, H, W, C]
                truth_np = np.transpose(raw_truth, (1, 2, 0, 3))               # [H, W, T, C]

                _, _, t_pred, _ = pred_np.shape
                t_export = min(t_pred, truth_np.shape[2], n_timesteps)

                print(f"[export_vtk] Sample {sample_count} (seed={seed_key}): exporting {t_export} timesteps")

                # Export each timestep
                for t in range(t_export):
                    # Transpose for VTK: [H, W] -> [W, H] (imshow convention)
                    truth_field = truth_np[:, :, t, 0].T  # [Ny=W, Nx=H]
                    pred_field = pred_np[:, :, t, 0].T

                    truth_file = vtk_dir / f"sample{sample_count:02d}_truth_t{t:04d}.vtk"
                    pred_file = vtk_dir / f"sample{sample_count:02d}_pred_t{t:04d}.vtk"

                    write_vtk_structured(truth_field, x_min, x_max, y_min, y_max, truth_file, "h")
                    write_vtk_structured(pred_field, x_min, x_max, y_min, y_max, pred_file, "h")

                sample_count += 1
                total_files += t_export * 2

    print(f"[export_vtk] Exported {sample_count} samples to {vtk_dir}")
    print(f"[export_vtk] Total VTK files: {total_files} (truth + pred)")


def parse_args():
    p = argparse.ArgumentParser(description="Export model predictions to VTK format")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--model", default="unet_lomix", 
                   help="Model to export (unet, unet_lomix, fno, or 'all')")
    p.add_argument("--n_samples", type=int, default=1,
                   help="Number of test samples to export")
    p.add_argument("--n_timesteps", type=int, default=101,
                   help="Number of timesteps to export per sample")
    return p.parse_args()


def main():
    args = parse_args()
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg["data"]["base_path"] = str(ROOT / "../data/")
    
    if args.model == "all":
        models = ["unet", "unet_lomix", "fno"]
    else:
        models = [args.model]
    
    print(f"[export_vtk] Exporting VTK files for models: {models}")
    print(f"[export_vtk] Samples: {args.n_samples}  Timesteps: {args.n_timesteps}")
    
    for model_name in models:
        export_model_vtk(
            model_name,
            cfg,
            n_samples=args.n_samples,
            n_timesteps=args.n_timesteps,
        )
    
    print(f"\n[export_vtk] Complete! VTK files saved to: {ROOT}/results/vtk/")
    print("[export_vtk] Open with ParaView or VisIt for visualization.")


if __name__ == "__main__":
    main()

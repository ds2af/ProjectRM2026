"""
scripts/export_vtk.py
=====================
Export ground truth and model predictions to VTK format for visualization.

Usage
-----
    python scripts/export_vtk.py --model unet_lomix --config configs/default.yaml
    python scripts/export_vtk.py --model fno --n_samples 1 --n_timesteps 50

VTK files saved to: results/vtk/<model>/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import SWEDataset, SWEDatasetWithGrid
from src.models.unet import UNet2d, UNet2dLoMix
from src.models.fno import FNO2d


def write_vtk_structured(
    field: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    filename: Path,
    var_name: str = "h"
) -> None:
    """Write 2D scalar field to VTK STRUCTURED_POINTS format."""
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
) -> None:
    """Export VTK files for a specific trained model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[export_vtk] Model: {model_name}  Device: {device}")
    
    # Load checkpoint
    checkpoint_path = ROOT / cfg["paths"]["checkpoint_dir"] / f"{model_name}_best.pt"
    if not checkpoint_path.exists():
        print(f"[export_vtk] Checkpoint not found: {checkpoint_path}")
        return
    
    # Build test dataset/loader
    needs_grid = (model_name == "fno")
    ds_cls = SWEDatasetWithGrid if needs_grid else SWEDataset
    test_ds = ds_cls(
        filename=cfg["data"]["filename"],
        saved_folder=cfg["data"]["base_path"],
        initial_step=cfg["data"]["initial_step"],
        if_test=True,
        test_ratio=cfg["data"].get("test_ratio", 0.1),
        val_ratio=cfg["data"].get("val_ratio", 0.0),
        max_samples=n_samples,
    )
    
    # Load model
    sample_xx = test_ds[0][0].unsqueeze(0)
    if model_name == "unet":
        model = UNet2d.from_config(cfg, sample_xx, use_lomix=False).to(device)
    elif model_name == "unet_lomix":
        model = UNet2dLoMix.from_config(cfg, sample_xx, use_lomix=True).to(device)
    elif model_name == "fno":
        model = FNO2d.from_config(cfg, sample_xx).to(device)
    else:
        print(f"[export_vtk] Unknown model: {model_name}")
        return
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    print(f"[export_vtk] Loaded checkpoint: {checkpoint_path}")
    
    # Export VTK
    vtk_dir = ROOT / "results" / "vtk" / model_name
    vtk_dir.mkdir(parents=True, exist_ok=True)
    
    ini = cfg["data"]["initial_step"]
    n_exported = 0
    
    with torch.no_grad():
        for sample_idx in range(min(n_samples, len(test_ds))):
            if needs_grid:
                xx, yy, grid = test_ds[sample_idx]
                xx, yy, grid = xx.unsqueeze(0).to(device), yy.unsqueeze(0).to(device), grid.unsqueeze(0).to(device)
            else:
                xx, yy = test_ds[sample_idx]
                xx, yy = xx.unsqueeze(0).to(device), yy.unsqueeze(0).to(device)
            
            # Export ground truth
            for t in range(min(n_timesteps, yy.shape[-2])):
                field = yy[0, :, :, t, 0].cpu().numpy()
                out_path = vtk_dir / f"sample{sample_idx:02d}_truth_t{t:04d}.vtk"
                write_vtk_structured(field, -2.5, 2.5, -2.5, 2.5, out_path, "h_truth")
            
            # Export predictions
            pred = yy[..., :ini, :]
            inp_shp = list(xx.shape[:-2]) + [-1]
            for t in range(ini, min(n_timesteps, yy.shape[-2])):
                inp = xx.reshape(inp_shp)
                if needs_grid:
                    im = model(inp, grid)
                else:
                    inp_ch = inp.permute(0, -1, *range(1, len(inp_shp) - 1))
                    out = model(inp_ch)
                    im = out.permute(0, *range(2, out.ndim), 1).unsqueeze(-2)
                
                field = im[0, :, :, 0, 0].cpu().numpy()
                out_path = vtk_dir / f"sample{sample_idx:02d}_pred_t{t:04d}.vtk"
                write_vtk_structured(field, -2.5, 2.5, -2.5, 2.5, out_path, "h_pred")
                
                pred = torch.cat([pred, im], dim=-2)
                xx   = torch.cat([xx[..., 1:, :], im], dim=-2)
            
            n_exported += 1
            print(f"[export_vtk] Exported sample {sample_idx}: {n_exported} trajectories")
    
    print(f"[export_vtk] Total VTK files exported: {n_exported * min(n_timesteps, 101) * 2}")


def parse_args():
    p = argparse.ArgumentParser(description="Export VTK files for trained models")
    p.add_argument("--model", default="unet_lomix", choices=["unet", "unet_lomix", "fno"])
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--n_samples", type=int, default=1)
    p.add_argument("--n_timesteps", type=int, default=101)
    return p.parse_args()


def main():
    args = parse_args()
    with open(ROOT / args.config) as f:
        cfg = yaml.safe_load(f)
    cfg["data"]["base_path"] = str(ROOT / "../data/")
    
    export_model_vtk(args.model, cfg, n_samples=args.n_samples, n_timesteps=args.n_timesteps)
    print("[export_vtk] Done.")


if __name__ == "__main__":
    main()

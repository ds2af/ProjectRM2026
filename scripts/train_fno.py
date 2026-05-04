"""
scripts/train_fno.py
====================
Training script for the Fourier Neural Operator (FNO2d).
Uses the SWEDatasetWithGrid loader which returns (xx, yy, grid).

AI Disclaimer
-------------
Coding assistance from ChatGPT and GitHub Copilot was used during development.
The author has thoroughly reviewed, checked, and verified the code for correctness
and takes responsibility for the final implementation used in this project.

Usage
-----
    python scripts/train_fno.py --config configs/default.yaml
    python scripts/train_fno.py --config configs/default.yaml --quick
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import build_dataloaders
from src.models.fno import FNO2d
from src.utils.logger import ExperimentLogger
from src.utils.metrics import compute_all_metrics
from src.utils.trainer import Trainer


def parse_args():
    p = argparse.ArgumentParser(description="Train FNO2d surrogate")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    p.add_argument("--resume", action="store_true", help="Resume from rolling last checkpoint if available")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _sync_if_cuda(device: torch.device) -> None:
    """Synchronize CUDA stream for stable wall-clock timing."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def run(cfg, quick=False, epoch_override=None, resume=False, seed=None):
    if seed is not None:
        set_seed(int(seed))
        print(f"[train_fno] Seed: {seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_fno] Device: {device}")

    max_samples = -1
    if quick:
        q = cfg.get("quick", {})
        max_samples               = q.get("max_samples", 20)
        cfg["training"]["epochs"] = q.get("epochs", 3)
        cfg["training"]["batch_size"] = q.get("batch_size", 2)
        print(f"[train_fno] Quick mode: {max_samples} samples, {cfg['training']['epochs']} epochs")
    if epoch_override is not None:
        cfg["training"]["epochs"] = epoch_override

    # FNO needs grid → use_grid=True
    train_loader, val_loader, test_loader = build_dataloaders(cfg, with_grid=True, max_samples=max_samples)

    sample_xx, _, _ = next(iter(train_loader))
    model = FNO2d.from_config(cfg, sample_xx).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train_fno] Parameters: {n_params:,}")

    tr = cfg["training"]
    optimizer = torch.optim.Adam(model.parameters(), lr=tr["learning_rate"], weight_decay=tr["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=tr["scheduler_step"], gamma=tr["scheduler_gamma"])

    log_path = ROOT / cfg["paths"]["log_dir"] / "fno_log.csv"
    logger = ExperimentLogger(
        "fno",
        log_dir=str(ROOT / cfg["paths"]["log_dir"]),
        results_dir=str(ROOT / cfg["paths"]["results_dir"]),
        append=bool(resume and log_path.exists()),
    )
    trainer = Trainer("fno", model, optimizer, scheduler, torch.nn.MSELoss(), device, cfg, logger, str(ROOT / cfg["paths"]["checkpoint_dir"]))

    start_epoch = 0
    if resume:
        ckpt_dir = ROOT / cfg["paths"]["checkpoint_dir"]
        ckpt_last = ckpt_dir / "fno_last.pt"
        ckpt_best = ckpt_dir / "fno_best.pt"
        ckpt_to_use = ckpt_last if ckpt_last.exists() else (ckpt_best if ckpt_best.exists() else None)
        if ckpt_to_use is not None:
            loaded_epoch = trainer.load_checkpoint(ckpt_to_use)
            start_epoch = int(loaded_epoch) + 1
            print(f"[train_fno] Resuming from {ckpt_to_use} at epoch {start_epoch}")
        else:
            print("[train_fno] Resume requested but no checkpoint found; starting from scratch.")

    mcfg  = cfg.get("fno", {})
    unroll  = mcfg.get("unroll_step", 20)
    t_train = cfg["data"].get("t_train", 40)

    print("[train_fno] Starting autoregressive training …")
    trainer.fit(
        train_loader,
        val_loader,
        ar_mode=True,
        t_train=t_train,
        unroll_step=unroll,
        use_pushforward=False,
        start_epoch=start_epoch,
    )

    # Evaluate the same model snapshot used by checkpoint-based plotting.
    best_ckpt = ROOT / cfg["paths"]["checkpoint_dir"] / "fno_best.pt"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
        print(f"[train_fno] Loaded best checkpoint for evaluation: {best_ckpt}")
    else:
        print(f"[train_fno] Warning: best checkpoint not found, evaluating current model: {best_ckpt}")

    # ── Evaluate ────────────────────────────────────────────────────────
    ini = cfg["data"]["initial_step"]
    model.eval()
    all_pred, all_tgt = [], []
    _sync_if_cuda(device)
    inf_start = time.perf_counter()
    with torch.no_grad():
        for xx, yy, grid in test_loader:
            xx, yy, grid = xx.to(device), yy.to(device), grid.to(device)
            pred = yy[..., :ini, :]
            inp_shp = list(xx.shape[:-2]) + [-1]
            for t in range(ini, yy.shape[-2]):
                inp_flat = xx.reshape(inp_shp)                      # [B, H, W, T*C]
                im = model(inp_flat, grid)                           # [B, H, W, 1, C]
                pred = torch.cat([pred, im], dim=-2)
                xx   = torch.cat([xx[..., 1:, :], im], dim=-2)
            all_pred.append(pred.cpu())
            all_tgt.append(yy.cpu())
    _sync_if_cuda(device)
    inf_elapsed = time.perf_counter() - inf_start

    pred_cat   = torch.cat(all_pred)
    target_cat = torch.cat(all_tgt)
    metrics = compute_all_metrics(pred_cat, target_cat, initial_step=ini)
    metrics["inference_time_s"] = round(inf_elapsed, 4)
    metrics["n_params"]         = n_params
    logger.save_metrics(metrics)
    logger.close()
    print(f"[train_fno] Done. RMSE={metrics['rmse']:.4e}  relL2={metrics['rel_l2']:.4e}")


if __name__ == "__main__":
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg["data"]["base_path"] = str(ROOT / "../data/")
    run(cfg, quick=args.quick, epoch_override=args.epochs, resume=args.resume, seed=args.seed)

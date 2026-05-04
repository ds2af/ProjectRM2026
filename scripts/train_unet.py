"""
scripts/train_unet.py
====================
Training script for the U-Net (vanilla or LoMix variant) surrogate model,
using autoregressive pushforward rollout.

AI Disclaimer
-------------
Coding assistance from ChatGPT and GitHub Copilot was used during development.
The author has thoroughly reviewed, checked, and verified the code for correctness
and takes responsibility for the final implementation used in this project.

Usage
-----
    python scripts/train_unet.py --config configs/default.yaml
    python scripts/train_unet.py --config configs/default.yaml --quick
    python scripts/train_unet.py --config configs/default.yaml --vanilla  # UNet2d only
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
from src.models.unet import UNet2d, UNet2dLoMix
from src.utils.logger import ExperimentLogger
from src.utils.metrics import compute_all_metrics
from src.utils.trainer import Trainer


def parse_args():
    p = argparse.ArgumentParser(description="Train U-Net surrogate (AR pushforward)")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    p.add_argument("--vanilla", action="store_true", help="Use plain UNet2d (not LoMix)")
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


def _try_load_resume_checkpoint(trainer: Trainer, checkpoint_path: Path) -> int:
    """Load a checkpoint if it is readable and return the saved epoch."""
    loaded_epoch = trainer.load_checkpoint(checkpoint_path)
    return int(loaded_epoch)


def run(cfg, quick=False, epoch_override=None, use_lomix=True, seed=None, resume=False):
    if seed is not None:
        set_seed(int(seed))
        print(f"[train_unet] Seed: {seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_tag = "unet_lomix" if use_lomix else "unet"
    print(f"[train_unet] Device: {device}  model: {model_tag}")

    max_samples = -1
    if quick:
        q = cfg.get("quick", {})
        max_samples               = q.get("max_samples", 20)
        cfg["training"]["epochs"] = q.get("epochs", 3)
        cfg["training"]["batch_size"] = q.get("batch_size", 2)
        print(f"[train_unet] Quick mode: {max_samples} samples, {cfg['training']['epochs']} epochs")
    if epoch_override is not None:
        cfg["training"]["epochs"] = epoch_override

    train_loader, val_loader, test_loader = build_dataloaders(cfg, max_samples=max_samples)

    sample_xx, _ = next(iter(train_loader))
    B, H, W, T, C = sample_xx.shape
    mcfg = cfg.get("unet", {})
    init_feat = mcfg.get("init_features", 32)
    dropout   = mcfg.get("dropout", 0.0)
    ModelCls  = UNet2dLoMix if use_lomix else UNet2d
    model     = ModelCls(in_channels=C * T, out_channels=C, init_features=init_feat, dropout=dropout).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train_unet] Parameters: {n_params:,}")

    tr = cfg["training"]
    optimizer = torch.optim.Adam(model.parameters(), lr=tr["learning_rate"], weight_decay=tr["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=tr["scheduler_step"], gamma=tr["scheduler_gamma"])

    log_path = ROOT / cfg["paths"]["log_dir"] / f"{model_tag}_log.csv"
    logger = ExperimentLogger(
        model_tag,
        log_dir=str(ROOT / cfg["paths"]["log_dir"]),
        results_dir=str(ROOT / cfg["paths"]["results_dir"]),
        append=bool(resume and log_path.exists()),
    )
    trainer = Trainer(model_tag, model, optimizer, scheduler, torch.nn.MSELoss(), device, cfg, logger, str(ROOT / cfg["paths"]["checkpoint_dir"]))

    start_epoch = 0
    if resume:
        ckpt_dir = ROOT / cfg["paths"]["checkpoint_dir"]
        ckpt_last = ckpt_dir / f"{model_tag}_last.pt"
        ckpt_best = ckpt_dir / f"{model_tag}_best.pt"
        if ckpt_last.exists():
            try:
                loaded_epoch = _try_load_resume_checkpoint(trainer, ckpt_last)
                start_epoch = loaded_epoch + 1
                print(f"[train_unet] Resuming from {ckpt_last} at epoch {start_epoch}")
            except Exception as exc:
                print(f"[train_unet] Warning: failed to load {ckpt_last}: {exc}")
                if ckpt_best.exists():
                    loaded_epoch = _try_load_resume_checkpoint(trainer, ckpt_best)
                    start_epoch = loaded_epoch + 1
                    print(f"[train_unet] Falling back to {ckpt_best} at epoch {start_epoch}")
                else:
                    print("[train_unet] Resume requested but no readable checkpoint found; starting from scratch.")
        elif ckpt_best.exists():
            loaded_epoch = _try_load_resume_checkpoint(trainer, ckpt_best)
            start_epoch = loaded_epoch + 1
            print(f"[train_unet] Resuming from {ckpt_best} at epoch {start_epoch}")
        else:
            print("[train_unet] Resume requested but no checkpoint found; starting from scratch.")

    unroll = mcfg.get("unroll_step", 20)
    t_train = cfg["data"].get("t_train", 40)

    print("[train_unet] Starting autoregressive training …")
    trainer.fit(
        train_loader,
        val_loader,
        ar_mode=True,
        t_train=t_train,
        unroll_step=unroll,
        use_pushforward=True,
        start_epoch=start_epoch,
    )

    # Evaluate the same model snapshot used by downstream figures/checkpoint loads.
    best_ckpt = ROOT / cfg["paths"]["checkpoint_dir"] / f"{model_tag}_best.pt"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
        print(f"[train_unet] Loaded best checkpoint for evaluation: {best_ckpt}")
    else:
        print(f"[train_unet] Warning: best checkpoint not found, evaluating current model: {best_ckpt}")

    # ── Evaluate ────────────────────────────────────────────────────────
    ini = cfg["data"]["initial_step"]
    model.eval()
    all_pred, all_tgt = [], []
    _sync_if_cuda(device)
    inf_start = time.perf_counter()
    with torch.no_grad():
        for xx, yy in test_loader:
            xx, yy = xx.to(device), yy.to(device)
            pred = yy[..., :ini, :]
            inp_shp = list(xx.shape[:-2]) + [-1]
            for t in range(ini, yy.shape[-2]):
                inp = xx.reshape(inp_shp).permute(0, -1, *range(1, len(inp_shp) - 1))
                out = model(inp)
                im = out.permute(0, *range(2, out.ndim), 1).unsqueeze(-2)
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
    print(f"[train_unet] Done. RMSE={metrics['rmse']:.4e}  inf={metrics['inference_time_s']:.4f}s")


if __name__ == "__main__":
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg["data"]["base_path"] = str(ROOT / "../data/")
    run(
        cfg,
        quick=args.quick,
        epoch_override=args.epochs,
        use_lomix=not args.vanilla,
        seed=args.seed,
        resume=args.resume,
    )

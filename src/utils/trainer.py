"""
src/utils/trainer.py
=====================
Generic training loop shared by U-Net and FNO.

The trainer:
* Supports both single-step and autoregressive rollout modes.
* Decouples model-specific forward logic via a ``step_fn`` callback.
  – ``step_fn(model, xx, yy, device) → pred, loss``  (for non-AR models)
  – AR rollout is implemented inside the trainer for U-Net / FNO.
* Saves best checkpoint (lowest validation loss) automatically.
"""

from __future__ import annotations

import time
from pathlib import Path
from timeit import default_timer
from typing import Callable, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.utils.logger import ExperimentLogger
from src.utils.metrics import compute_all_metrics


class Trainer:
    """
    Generic trainer.

    Parameters
    ----------
    model_name     : str
    model          : nn.Module
    optimizer      : torch.optim.Optimizer
    scheduler      : LR scheduler or None
    loss_fn        : callable(pred, target) → scalar tensor
    device         : torch.device
    cfg            : dict  (full config dict)
    logger         : ExperimentLogger
    checkpoint_dir : Path  (where ``.pt`` checkpoints are saved)
    """

    def __init__(
        self,
        model_name: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        loss_fn: Callable,
        device: torch.device,
        cfg: dict,
        logger: ExperimentLogger,
        checkpoint_dir: str = "results/checkpoints",
    ) -> None:
        self.model_name     = model_name
        self.model          = model
        self.optimizer      = optimizer
        self.scheduler      = scheduler
        self.loss_fn        = loss_fn
        self.device         = device
        self.cfg            = cfg
        self.logger         = logger
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        tr_cfg              = cfg["training"]
        self.epochs         = tr_cfg["epochs"]
        self.model_update   = tr_cfg.get("model_update_freq", 5)
        self.grad_clip      = tr_cfg.get("gradient_clip", 0.0)
        self.early_stop_patience = int(tr_cfg.get("early_stopping_patience", 0) or 0)
        self.early_stop_min_delta = float(tr_cfg.get("early_stopping_min_delta", 0.0) or 0.0)
        self.initial_step   = cfg["data"]["initial_step"]

        self.best_val_loss  = float("inf")
        self.train_losses: list[float] = []
        self.val_losses: list[float]   = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        step_fn: Optional[Callable] = None,
        ar_mode: bool = False,
        t_train: Optional[int] = None,
        unroll_step: int = 20,
        use_pushforward: bool = True,
        start_epoch: int = 0,
    ) -> None:
        """
        Run training for ``self.epochs`` epochs.

        Parameters
        ----------
        train_loader, val_loader : DataLoader | None
        step_fn : callable  ``(model, xx, yy, device) → (pred, loss)``
            For single-step models.  If None and ar_mode is False, a
            default flatten-and-forward step is used.
        ar_mode : bool
            If True, use the autoregressive pushforward rollout.
        t_train : int or None
            Total time steps to unroll in AR mode.
        unroll_step : int
            Pushforward unroll window size.
        use_pushforward : bool
            If True, use PDEBench-style pushforward trick in training
            (no-grad rollout before final unroll window). If False,
            accumulate loss on all rollout steps.
        start_epoch : int
            Epoch index to start from (used for crash-resume).
        """
        if start_epoch >= self.epochs:
            print(
                f"  [{self.model_name}] start_epoch={start_epoch} >= epochs={self.epochs}; "
                "skipping further training."
            )
            return

        bad_epochs = 0

        for epoch in range(start_epoch, self.epochs):
            t_start = default_timer()
            train_loss = self._train_epoch(
                train_loader, step_fn, ar_mode, t_train, unroll_step, use_pushforward
            )
            t_end = default_timer()

            val_loss = train_loss
            has_validation = val_loader is not None and len(val_loader) > 0
            should_validate = has_validation and (
                self.early_stop_patience > 0
                or epoch % self.model_update == 0
                or epoch == self.epochs - 1
            )
            if should_validate:
                if has_validation:
                    val_loss = self._val_epoch(
                        val_loader, step_fn, ar_mode, t_train, use_pushforward
                    )
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(epoch, val_loss)
                    bad_epochs = 0
                elif self.early_stop_patience > 0:
                    bad_epochs += 1

            if self.scheduler is not None:
                self.scheduler.step()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.logger.log_epoch(epoch, train_loss, val_loss)
            self.save_last_checkpoint(epoch, val_loss)

            if epoch % max(1, self.epochs // 10) == 0:
                print(
                    f"  [{self.model_name}] epoch {epoch:4d}/{self.epochs}  "
                    f"train={train_loss:.4e}  val={val_loss:.4e}  "
                    f"t={t_end - t_start:.1f}s"
                )

            if self.early_stop_patience > 0 and has_validation and bad_epochs >= self.early_stop_patience:
                print(
                    f"  [{self.model_name}] early stopping after {bad_epochs} validation epochs "
                    f"without improvement (patience={self.early_stop_patience})."
                )
                break

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train_epoch(self, loader, step_fn, ar_mode, t_train, unroll_step, use_pushforward):
        self.model.train()
        total_loss = 0.0
        n_batches  = 0

        for batch in loader:
            self.optimizer.zero_grad()

            if ar_mode:
                loss = self._ar_step(batch, t_train, unroll_step, train=True, use_pushforward=use_pushforward)
            elif step_fn is not None:
                xx, yy = batch[0].to(self.device), batch[1].to(self.device)
                _, loss = step_fn(self.model, xx, yy, self.device)
            else:
                loss = self._default_step(batch)

            loss.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        return total_loss / max(n_batches, 1)

    def _val_epoch(self, loader, step_fn, ar_mode, t_train, use_pushforward):
        self.model.eval()
        total_loss = 0.0
        n_batches  = 0

        with torch.no_grad():
            for batch in loader:
                if ar_mode:
                    loss = self._ar_step(batch, t_train, t_train, train=False, use_pushforward=use_pushforward)
                elif step_fn is not None:
                    xx, yy = batch[0].to(self.device), batch[1].to(self.device)
                    _, loss = step_fn(self.model, xx, yy, self.device)
                else:
                    loss = self._default_step(batch)

                total_loss += loss.item()
                n_batches  += 1

        return total_loss / max(n_batches, 1)

    def _default_step(self, batch) -> torch.Tensor:
        """
        Fallback single-step forward for flat models.
        Predicts the *next* step from the context window.
        """
        xx, yy = batch[0].to(self.device), batch[1].to(self.device)
        # Context: [B, H, W, initial_step, C]
        # Target: next step [B, H, W, 1, C]
        target = yy[..., self.initial_step : self.initial_step + 1, :]
        pred   = self.model(xx)  # shapes handled inside model
        _b = target.shape[0]
        return self.loss_fn(pred.reshape(_b, -1), target.reshape(_b, -1))

    def _ar_step(self, batch, t_train, unroll_step, train: bool, use_pushforward: bool = True) -> torch.Tensor:
        """
        Autoregressive pushforward step used by U-Net and FNO.

        batch may be (xx, yy) or (xx, yy, grid).
        """
        has_grid = len(batch) == 3
        xx = batch[0].to(self.device)
        yy = batch[1].to(self.device)
        grid = batch[2].to(self.device) if has_grid else None

        if t_train is None:
            t_train = yy.shape[-2]
        t_train = min(t_train, yy.shape[-2])

        if t_train - unroll_step < 1:
            unroll_step = t_train - 1

        loss = torch.tensor(0.0, device=self.device)
        pred = yy[..., : self.initial_step, :]

        inp_shape = list(xx.shape[:-2]) + [-1]  # flatten time*channel

        def _rollout_once(inp_flat: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
            if has_grid:
                return self.model(inp_flat, grid)  # FNO

            # [B, H, W, t*C] -> [B, t*C, H, W]
            inp_ch = inp_flat.permute(0, -1, *range(1, inp_flat.ndim - 1))
            out = self.model(inp_ch)
            im = out.permute(0, *range(2, out.ndim), 1).unsqueeze(-2)
            return im

        for t in range(self.initial_step, t_train):
            inp = xx.reshape(inp_shape)

            # Match PDEBench pushforward training:
            # do no-grad rollout before the final unroll window that contributes to loss.
            use_no_grad = train and use_pushforward and (t < t_train - unroll_step)
            if use_no_grad:
                with torch.no_grad():
                    im = _rollout_once(inp, xx)
            else:
                im = _rollout_once(inp, xx)

            y  = yy[..., t : t + 1, :]
            _b = y.shape[0]

            # Reference behavior:
            # - pushforward=True  -> backprop only on the final unroll window
            # - pushforward=False -> backprop on all rollout steps
            if (not use_pushforward) or (t >= t_train - unroll_step):
                loss = loss + self.loss_fn(im.reshape(_b, -1), y.reshape(_b, -1))

            pred = torch.cat([pred, im], dim=-2)
            xx   = torch.cat([xx[..., 1:, :], im], dim=-2)

        return loss

    # ------------------------------------------------------------------
    def save_checkpoint(self, epoch: int, val_loss: float) -> Path:
        path = self.checkpoint_dir / f"{self.model_name}_best.pt"
        torch.save(
            {
                "epoch":               epoch,
                "model_state_dict":    self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss":            val_loss,
            },
            path,
        )
        print(f"  [Trainer] Checkpoint saved -> {path}  (val_loss={val_loss:.4e})")
        return path

    def save_last_checkpoint(self, epoch: int, val_loss: float) -> Path:
        """Save rolling latest-state checkpoint for robust crash-resume."""
        path = self.checkpoint_dir / f"{self.model_name}_last.pt"
        torch.save(
            {
                "epoch":               epoch,
                "model_state_dict":    self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss":            val_loss,
            },
            path,
        )
        return path

    def load_checkpoint(self, path: str | Path) -> int:
        """Load from path; returns epoch number."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.best_val_loss = ckpt.get("val_loss", float("inf"))
        return ckpt.get("epoch", 0)

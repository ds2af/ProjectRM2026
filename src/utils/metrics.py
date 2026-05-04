"""
src/utils/metrics.py
====================
Evaluation metrics used uniformly across all five model types.

AI Disclaimer
-------------
Coding assistance from ChatGPT and GitHub Copilot was used during development.
The author has thoroughly reviewed, checked, and verified the code for correctness
and takes responsibility for the final implementation used in this project.

Functions
---------
rmse(pred, target)          → scalar RMSE over entire tensor
per_timestep_rmse(...)      → RMSE curve over time steps [T]
compute_all_metrics(...)    → dict with standard metrics
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Basic metrics (operate on arbitrary-shape tensors)
# ---------------------------------------------------------------------------

def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Root-mean-squared error averaged over all elements.

    Parameters
    ----------
    pred, target : torch.Tensor  (any shape, must match)

    Returns
    -------
    float
    """
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()


def per_timestep_rmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    time_dim: int = -2,
) -> torch.Tensor:
    """
    Compute RMSE at each individual time step.

    Parameters
    ----------
    pred, target : torch.Tensor  [B, H, W, T, C]  (or similar)
    time_dim     : int            which dimension is time

    Returns
    -------
    rmse_curve : FloatTensor [T]
    """
    sq_err = (pred - target) ** 2
    # Mean over all dims except time_dim
    dims = list(range(pred.ndim))
    dims.remove(time_dim % pred.ndim)
    mean_sq = sq_err.mean(dim=dims)
    return torch.sqrt(mean_sq)


def compute_all_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    initial_step: int = 10,
) -> dict:
    """
    Compute the full metric suite used in the comparative evaluation.

    The ``initial_step`` timesteps fed as context input are excluded
    from the error calculation (we only penalize predicted steps).

    Parameters
    ----------
    pred         : FloatTensor [B, H, W, T, C]
    target       : FloatTensor [B, H, W, T, C]
    initial_step : int

    Returns
    -------
    dict with key: rmse
    """
    # Exclude conditioning steps
    p = pred[..., initial_step:, :]
    t = target[..., initial_step:, :]

    return {"rmse": rmse(p, t)}

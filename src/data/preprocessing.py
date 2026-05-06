"""
src/data/preprocessing.py
==========================
Normalization utilities shared across all model experiments.

All models operate on the same channel-wise z-score normalized data so that
results are directly comparable.  Normalization statistics are computed from
the *training* split only and then applied consistently to validation and test.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import DataLoader


def compute_normalization_stats(
    loader: DataLoader,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-channel mean and standard deviation from a DataLoader.

    Iterates over the full loader once, streaming batches.

    Parameters
    ----------
    loader : DataLoader
        Expected to yield tuples ``(xx, yy)`` or ``(xx, yy, grid)``.
        Mean / std are computed over the *full trajectory* ``yy``.

    Returns
    -------
    mean : FloatTensor [C]
    std  : FloatTensor [C]
    """
    # Accumulate online (Welford not needed for batch-level sums)
    total_sum = None
    total_sq  = None
    total_n   = 0

    for batch in loader:
        yy = batch[1]  # [B, H, W, T, C]
        B = yy.shape[0]
        # Flatten everything except the channel dimension
        # yy reshaped → [B * H * W * T, C]
        flat = yy.reshape(-1, yy.shape[-1])
        if total_sum is None:
            C = flat.shape[-1]
            total_sum = torch.zeros(C)
            total_sq  = torch.zeros(C)
        total_sum += flat.sum(dim=0)
        total_sq  += (flat ** 2).sum(dim=0)
        total_n   += flat.shape[0]

    mean = total_sum / total_n
    std  = torch.sqrt(total_sq / total_n - mean ** 2).clamp(min=1e-8)
    return mean, std


class Normalizer:
    """
    Stores mean and std tensors and provides normalize / denormalize helpers.

    Parameters
    ----------
    mean : FloatTensor [C]
    std  : FloatTensor [C]
    device : torch.device, optional
    """

    def __init__(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> None:
        self.mean = mean
        self.std = std
        if device is not None:
            self.to(device)

    def to(self, device: torch.device) -> "Normalizer":
        self.mean = self.mean.to(device)
        self.std  = self.std.to(device)
        return self

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize x (last dim = C) using stored stats."""
        return (x - self.mean) / self.std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Reverse normalization."""
        return x * self.std + self.mean

    def state_dict(self) -> dict:
        return {"mean": self.mean.cpu(), "std": self.std.cpu()}

    @classmethod
    def from_state_dict(cls, d: dict) -> "Normalizer":
        return cls(d["mean"], d["std"])

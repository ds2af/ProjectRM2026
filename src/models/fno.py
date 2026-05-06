"""
src/models/fno.py
==================
Fourier Neural Operator (FNO2d) adapted for the 2-D SWE/PDE benchmark.

Source
------
Based on Fourier Neural Operator implementations from PDEBench,
which derives from Zongyi Li et al. (2020), MIT License.

Architecture
------------
The FNO2d applies 4 Fourier integral operator layers in a residual fashion:

    u'(x) = W u(x) + ∫ κ(x-y) u(y) dy

where the integral operator is evaluated cheaply in Fourier space by
truncating to ``modes1 × modes2`` Fourier coefficients.

Input convention
----------------
    x    : FloatTensor [B, H, W, C*t]   spatial-first, channels last
    grid : FloatTensor [B, H, W, 2]     normalized (x, y) coordinates in [0,1]

Both are concatenated along the last dimension before the lifting layer fc0.

Output
------
    FloatTensor [B, H, W, 1, C]   (next time-step prediction)

References
----------
Z. Li, N. Kovachki, K. Azizzadenesheli, et al.  "Fourier Neural Operator for
Parametric Partial Differential Equations." ICLR 2021.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------
# Spectral (Fourier) convolution layer
# ---------------------------------------------------------------------------

class SpectralConv2d(nn.Module):
    """
    2-D real-valued spectral convolution (FNO Fourier layer).

    Computes: output = IFFT2[ R(k) · FFT2(input) ]
    where R(k) is a learnable complex weight tensor of shape
    [in_ch, out_ch, modes1, modes2].

    Parameters
    ----------
    in_channels, out_channels : int
    modes1, modes2            : int  number of Fourier modes kept (≤ floor(N/2)+1)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
    ) -> None:
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    @staticmethod
    def _cmul2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Batched complex matrix-vector product: (b,i,x,y),(i,o,x,y)→(b,o,x,y)."""
        return torch.einsum("bixy,ioxy->boxy", a, b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x)                        # [B, C, H, W//2+1]

        out_ft = torch.zeros(
            B, self.out_channels, H, W // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )
        # Top-left quadrant
        out_ft[:, :, : self.modes1, : self.modes2] = self._cmul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        # Bottom-left quadrant (negative x-frequencies)
        out_ft[:, :, -self.modes1 :, : self.modes2] = self._cmul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )
        return torch.fft.irfft2(out_ft, s=(H, W))        # [B, out_ch, H, W]


# ---------------------------------------------------------------------------
# FNO2d model
# ---------------------------------------------------------------------------

class FNO2d(nn.Module):
    """
    2-D Fourier Neural Operator.

    Parameters
    ----------
    num_channels  : int   number of field channels (C)
    modes1, modes2: int   number of Fourier modes per spatial dimension
    width         : int   internal channel width (lifting dimension)
    initial_step  : int   number of context time steps
    """

    def __init__(
        self,
        num_channels: int = 1,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 20,
        initial_step: int = 10,
    ) -> None:
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width  = width
        self.padding = 2  # pad non-periodic boundaries

        # Lifting layer: (C*t + 2) → width
        self.fc0 = nn.Linear(initial_step * num_channels + 2, width)

        # Four Fourier + point-wise skip residual layers
        self.conv0 = SpectralConv2d(width, width, modes1, modes2)
        self.conv1 = SpectralConv2d(width, width, modes1, modes2)
        self.conv2 = SpectralConv2d(width, width, modes1, modes2)
        self.conv3 = SpectralConv2d(width, width, modes1, modes2)
        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)

        # Projection layers: width → 128 → C
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, num_channels)

        self.num_channels = num_channels

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        grid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x    : FloatTensor [B, H, W, C*t]  (spatial-first, channels-last)
        grid : FloatTensor [B, H, W, 2]    normalized spatial coordinate

        Returns
        -------
        out  : FloatTensor [B, H, W, 1, C]
        """
        # ── Lifting ─────────────────────────────────────────────────────
        x = torch.cat([x, grid], dim=-1)   # [B, H, W, C*t+2]
        x = self.fc0(x)                    # [B, H, W, width]
        x = x.permute(0, 3, 1, 2)         # [B, width, H, W]

        x = F.pad(x, [0, self.padding, 0, self.padding])

        # ── Four Fourier operator layers ───────────────────────────────
        x = F.gelu(self.conv0(x) + self.w0(x))
        x = F.gelu(self.conv1(x) + self.w1(x))
        x = F.gelu(self.conv2(x) + self.w2(x))
        x = self.conv3(x) + self.w3(x)    # no activation before projection

        x = x[..., : -self.padding, : -self.padding]  # unpad
        x = x.permute(0, 2, 3, 1)         # [B, H, W, width]

        # ── Projection ──────────────────────────────────────────────────
        x = F.gelu(self.fc1(x))           # [B, H, W, 128]
        x = self.fc2(x)                   # [B, H, W, C]
        return x.unsqueeze(-2)            # [B, H, W, 1, C]

    # ------------------------------------------------------------------
    @staticmethod
    def from_config(cfg: dict, sample_xx: torch.Tensor) -> "FNO2d":
        B, H, W, T, C = sample_xx.shape
        mcfg = cfg.get("fno", {})
        return FNO2d(
            num_channels  = C,
            modes1        = mcfg.get("modes1", 12),
            modes2        = mcfg.get("modes2", 12),
            width         = mcfg.get("width", 20),
            initial_step  = T,
        )

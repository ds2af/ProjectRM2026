"""
src/models/unet.py
==================
Refactored U-Net implementations for 2-D spatiotemporal PDE surrogate modeling.

Source
------
Adapted and improved from U-Net implementations for medical image segmentation
(originally based on mateuszbuda/brain-segmentation-pytorch, MIT License).
Further refined for PDEBench 2-D shallow water equations benchmark.

Changes from original
---------------------
1. Optional dropout on encoder/decoder blocks.
2. Clean, self-contained docstrings; 1-D and 3-D variants removed for brevity
   (not needed in the 2-D SWE benchmark).
3. ``from_config`` class method for consistent construction from cfg dict.
4. ``UNet2dLoMix`` multi-scale fusion variant retained for reference comparison.

Forward convention (both variants)
------------------------------------
    Input  [B, C_in, H, W]   (channels-first spatial tensor)
    Output [B, C_out, H, W]
"""

from __future__ import annotations

from collections import OrderedDict
from typing import List

import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------
# Internal block helper
# ---------------------------------------------------------------------------

def _conv_block(in_ch: int, out_ch: int, name: str, dropout: float = 0.0) -> nn.Sequential:
    """
    Double-convolution block: Conv→BN→Tanh → Conv→BN→Tanh (+ optional Dropout).

    Tanh is kept to match the PDEBench U-Net reference implementation.
    """
    layers: list[tuple[str, nn.Module]] = [
        (
            name + "conv1",
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        ),
        (name + "norm1", nn.BatchNorm2d(out_ch)),
        (name + "act1", nn.Tanh()),
        (
            name + "conv2",
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        ),
        (name + "norm2", nn.BatchNorm2d(out_ch)),
        (name + "act2", nn.Tanh()),
    ]
    if dropout > 0.0:
        layers.append((name + "drop", nn.Dropout2d(dropout)))
    return nn.Sequential(OrderedDict(layers))


# ---------------------------------------------------------------------------
# UNet2d – vanilla encoder-decoder with skip connections
# ---------------------------------------------------------------------------

class UNet2d(nn.Module):
    """
    Standard 2-D U-Net with 4-level encoder-decoder and skip connections.

    Parameters
    ----------
    in_channels   : int  (C_in)  typically C * initial_step
    out_channels  : int  (C_out) typically C
    init_features : int  feature width at first encoder level (default 32)
    dropout       : float  Dropout2d probability (0 = disabled)
    """

    def __init__(
        self,
        in_channels: int = 10,
        out_channels: int = 1,
        init_features: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        f = init_features

        # Encoder
        self.enc1 = _conv_block(in_channels, f,      "enc1", dropout)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = _conv_block(f,      f * 2,  "enc2", dropout)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc3 = _conv_block(f * 2,  f * 4,  "enc3", dropout)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.enc4 = _conv_block(f * 4,  f * 8,  "enc4", dropout)
        self.pool4 = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = _conv_block(f * 8, f * 16, "bn", dropout)

        # Decoder
        self.up4   = nn.ConvTranspose2d(f * 16, f * 8,  2, stride=2)
        self.dec4  = _conv_block(f * 16, f * 8,  "dec4", dropout)
        self.up3   = nn.ConvTranspose2d(f * 8,  f * 4,  2, stride=2)
        self.dec3  = _conv_block(f * 8,  f * 4,  "dec3", dropout)
        self.up2   = nn.ConvTranspose2d(f * 4,  f * 2,  2, stride=2)
        self.dec2  = _conv_block(f * 4,  f * 2,  "dec2", dropout)
        self.up1   = nn.ConvTranspose2d(f * 2,  f,      2, stride=2)
        self.dec1  = _conv_block(f * 2,  f,      "dec1", dropout)

        # Output head
        self.head  = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        bn = self.bottleneck(self.pool4(e4))

        d4 = self.dec4(torch.cat([self.up4(bn), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.head(d1)

    @staticmethod
    def from_config(cfg: dict, sample_xx: torch.Tensor, use_lomix: bool = False) -> "UNet2d":
        B, H, W, T, C = sample_xx.shape
        mcfg = cfg.get("unet", {})
        cls = UNet2dLoMix if use_lomix else UNet2d
        return cls(
            in_channels   = C * T,
            out_channels  = C,
            init_features = mcfg.get("init_features", 32),
            dropout       = mcfg.get("dropout", 0.0),
        )


# ---------------------------------------------------------------------------
# UNet2dLoMix – multi-scale head fusion variant
# ---------------------------------------------------------------------------

class WeightedFusion2d(nn.Module):
    """
    LoMix-style per-pixel weighted fusion of multi-scale predictions.

    For each scale prediction ``P_s`` we learn a 1x1 conv that produces a
    spatial weight map. Weights are normalized with softmax across scales and
    used for weighted summation.
    """

    def __init__(self, num_scales: int, channels: int) -> None:
        super().__init__()
        self.weight_convs = nn.ModuleList(
            [nn.Conv2d(channels, 1, kernel_size=1) for _ in range(num_scales)]
        )

    def forward(self, preds: List[torch.Tensor]) -> torch.Tensor:
        """preds: list of [B, C, H, W] with the same shape."""
        weight_maps = [conv(pred) for pred, conv in zip(preds, self.weight_convs)]
        weights = torch.stack(weight_maps, dim=1)    # [B, S, 1, H, W]
        weights = torch.softmax(weights, dim=1)

        stacked = torch.stack(preds, dim=1)          # [B, S, C, H, W]
        return (weights * stacked).sum(dim=1)        # [B, C, H, W]

class UNet2dLoMix(nn.Module):
    """
    U-Net with LoMix-style multi-scale output mixing.

    Four side heads produce predictions at resolutions H/8, H/4, H/2, and H.
    All are upsampled to full resolution and fused with learned per-pixel
    weights via ``WeightedFusion2d``. This improves prediction at multiple
    spatial scales simultaneously.

    Parameters
    ----------
    Same as UNet2d.
    """

    def __init__(
        self,
        in_channels: int = 10,
        out_channels: int = 1,
        init_features: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        f = init_features

        # Encoder
        self.enc1 = _conv_block(in_channels, f,      "enc1", dropout)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = _conv_block(f,      f * 2,  "enc2", dropout)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc3 = _conv_block(f * 2,  f * 4,  "enc3", dropout)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.enc4 = _conv_block(f * 4,  f * 8,  "enc4", dropout)
        self.pool4 = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = _conv_block(f * 8, f * 16, "bn", dropout)

        # Decoder
        self.up4   = nn.ConvTranspose2d(f * 16, f * 8, 2, stride=2)
        self.dec4  = _conv_block(f * 16, f * 8, "dec4", dropout)
        self.up3   = nn.ConvTranspose2d(f * 8,  f * 4, 2, stride=2)
        self.dec3  = _conv_block(f * 8,  f * 4, "dec3", dropout)
        self.up2   = nn.ConvTranspose2d(f * 4,  f * 2, 2, stride=2)
        self.dec2  = _conv_block(f * 4,  f * 2, "dec2", dropout)
        self.up1   = nn.ConvTranspose2d(f * 2,  f,     2, stride=2)
        self.dec1  = _conv_block(f * 2,  f,     "dec1", dropout)

        # Multi-scale heads
        self.head4 = nn.Conv2d(f * 8, out_channels, 1)
        self.head3 = nn.Conv2d(f * 4, out_channels, 1)
        self.head2 = nn.Conv2d(f * 2, out_channels, 1)
        self.head1 = nn.Conv2d(f,     out_channels, 1)

        # LoMix fusion
        self.lomix = WeightedFusion2d(num_scales=4, channels=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        bn = self.bottleneck(self.pool4(e4))

        d4 = self.dec4(torch.cat([self.up4(bn), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        H, W = d1.shape[-2], d1.shape[-1]
        p1 = self.head1(d1)
        p2 = F.interpolate(self.head2(d2), (H, W), mode="bilinear", align_corners=False)
        p3 = F.interpolate(self.head3(d3), (H, W), mode="bilinear", align_corners=False)
        p4 = F.interpolate(self.head4(d4), (H, W), mode="bilinear", align_corners=False)

        return self.lomix([p1, p2, p3, p4])

    @staticmethod
    def from_config(cfg: dict, sample_xx: torch.Tensor, use_lomix: bool = False) -> "UNet2dLoMix":
        B, H, W, T, C = sample_xx.shape
        mcfg = cfg.get("unet", {})
        return UNet2dLoMix(
            in_channels   = C * T,
            out_channels  = C,
            init_features = mcfg.get("init_features", 32),
            dropout       = mcfg.get("dropout", 0.0),
        )

"""Encoders used by the segmentation branch.

SegEncoder: extracts multi-scale spatial priors Flq from the low-quality image.
MaskEncoder: encodes a ground-truth segmentation mask into a latent tensor.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(ch: int) -> nn.GroupNorm:
    num_groups = min(32, ch)
    while ch % num_groups != 0 and num_groups > 1:
        num_groups -= 1
    return nn.GroupNorm(num_groups=num_groups, num_channels=ch)


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            _norm(out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            _norm(out_ch),
            nn.SiLU(),
        )
        self.shortcut = (
            nn.Conv2d(in_ch, out_ch, 1)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.shortcut(x)


class SegEncoder(nn.Module):
    """CNN encoder that returns a feature map at the latent resolution.

    The paper uses multi-scale spatial priors Flq = Eseg(Xlq).  Here we return
    a single feature map of shape (B, out_channels, H/f, W/f) where f is the
    downsampling factor used by the VAE.  The architecture can be easily
    extended to return multiple intermediate scales if desired.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 32,
        base_channels: int = 32,
        downsample_steps: int = 4,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        layers = [nn.Conv2d(in_channels, base_channels, 3, padding=1), _norm(base_channels), nn.SiLU()]
        prev = base_channels
        for _ in range(downsample_steps):
            nxt = min(prev * 2, out_channels)
            layers.append(_ConvBlock(prev, nxt))
            layers.append(nn.AvgPool2d(2))
            prev = nxt
        layers.append(_ConvBlock(prev, out_channels))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MaskEncoder(nn.Module):
    """Encode a ground-truth segmentation mask into the same latent space as images.

    The mask can be multi-class (B, num_classes, H, W) or binary (B, 1, H, W).
    """

    def __init__(
        self,
        in_channels: int = 1,
        latent_channels: int = 4,
        base_channels: int = 32,
        downsample_steps: int = 4,
    ) -> None:
        super().__init__()
        layers = [nn.Conv2d(in_channels, base_channels, 3, padding=1), _norm(base_channels), nn.SiLU()]
        prev = base_channels
        for _ in range(downsample_steps):
            nxt = min(prev * 2, latent_channels * 8)
            layers.append(_ConvBlock(prev, nxt))
            layers.append(nn.AvgPool2d(2))
            prev = nxt
        layers.append(_ConvBlock(prev, latent_channels))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SegDecoder(nn.Module):
    """Decode a segmentation latent representation back to pixel-space logits."""

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        base_channels: int = 64,
        upsample_steps: int = 2,
    ) -> None:
        super().__init__()
        ch = in_channels
        layers = []
        for _ in range(upsample_steps):
            layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
            layers.append(nn.Conv2d(ch, base_channels, 3, padding=1))
            layers.append(_norm(base_channels))
            layers.append(nn.SiLU())
            ch = base_channels
        layers.append(nn.Conv2d(ch, out_channels, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

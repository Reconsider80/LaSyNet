"""Lightweight variational / deterministic autoencoder for the latent stage.

The paper builds LaSyNet on top of a frozen pretrained VAE (Evae / Dvae).
For this reference implementation we provide a small trainable autoencoder
with the same interface so that the project can run end-to-end.  In a real
setup you should replace it with a pretrained medical LDM first-stage such as
Stable-Diffusion's AutoencoderKL (CompVis/stable-diffusion-v1-4/vae) and
freeze it.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm_layer(channels: int) -> nn.GroupNorm:
    """GroupNorm that always works for the given channel count."""
    num_groups = min(32, channels)
    while channels % num_groups != 0 and num_groups > 1:
        num_groups -= 1
    return nn.GroupNorm(num_groups=num_groups, num_channels=channels)


class ResBlock(nn.Module):
    """Simple residual block with optional downsampling/upsampling."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.norm1 = _norm_layer(out_ch) if use_norm else nn.Identity()
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = _norm_layer(out_ch) if use_norm else nn.Identity()

        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride),
                _norm_layer(out_ch) if use_norm else nn.Identity(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        h = self.conv2(h)
        h = self.norm2(h)
        return self.act(h + self.shortcut(x))


class Autoencoder(nn.Module):
    """KL-regularized autoencoder (or deterministic if kl_weight=0).

    Args:
        in_channels: input channels (1 for medical grayscale).
        latent_channels: latent channel count (4 in LDM).
        base_channels: base width of the encoder/decoder.
        channel_mult: multipliers per downsampling level.
        num_res_blocks: residual blocks per level.
        kl_weight: weight of the KL term; set to 0.0 to train a deterministic AE.
        image_key: used by external wrappers to know input format.
    """

    def __init__(
        self,
        in_channels: int = 1,
        latent_channels: int = 4,
        base_channels: int = 64,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        kl_weight: float = 1e-6,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.kl_weight = kl_weight

        # Encoder
        self.encoder_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.encoder_blocks = nn.ModuleList()
        prev_ch = base_channels
        for i, mult in enumerate(channel_mult):
            ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.encoder_blocks.append(ResBlock(prev_ch, ch))
                prev_ch = ch
            if i != len(channel_mult) - 1:
                self.encoder_blocks.append(
                    nn.Sequential(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
                )

        self.encoder_out = nn.Conv2d(prev_ch, latent_channels * 2, 3, padding=1)

        # Decoder
        self.decoder_in = nn.Conv2d(latent_channels, prev_ch, 3, padding=1)
        self.decoder_blocks = nn.ModuleList()
        reversed_mult = list(reversed(channel_mult))
        for i, mult in enumerate(reversed_mult):
            ch = base_channels * mult
            for _ in range(num_res_blocks + 1):
                self.decoder_blocks.append(ResBlock(prev_ch, ch))
                prev_ch = ch
            if i != len(reversed_mult) - 1:
                self.decoder_blocks.append(
                    nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), ResBlock(prev_ch, prev_ch))
                )

        self.decoder_out = nn.Sequential(
            _norm_layer(prev_ch),
            nn.SiLU(),
            nn.Conv2d(prev_ch, in_channels, 3, padding=1),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (mu, logvar)."""
        h = self.encoder_in(x)
        for block in self.encoder_blocks:
            h = block(h)
        h = self.encoder_out(h)
        mu, logvar = h.split(self.latent_channels, dim=1)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training and self.kl_weight > 0:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_in(z)
        for block in self.decoder_blocks:
            h = block(h)
        return self.decoder_out(h)

    def forward(self, x: torch.Tensor) -> dict:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        loss = torch.tensor(0.0, device=x.device)
        if self.kl_weight > 0:
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1, 2, 3])
            loss = self.kl_weight * kl.mean()
        return {"recon": recon, "z": z, "mu": mu, "logvar": logvar, "loss": loss}

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic latent used by LaSyNet."""
        mu, _ = self.encode(x)
        return mu

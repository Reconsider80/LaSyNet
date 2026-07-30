"""Lightweight time-conditional UNet used as the frozen Rectified-Flow backbone.

The architecture follows the standard LDM-style UNet (Rombach et al. 2022)
but is kept compact and self-contained.  Adapters can be attached to every
ResBlock via the optional `adapters` argument of `forward`.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_timestep_embedding(
    timesteps: torch.Tensor, embedding_dim: int, scale: float = 1000.0
) -> torch.Tensor:
    """Sinusoidal timestep embedding for continuous t in [0, 1]."""
    if timesteps.dim() == 0:
        timesteps = timesteps.unsqueeze(0)
    timesteps = timesteps.view(-1) * scale
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.to(torch.float32)[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def _norm(ch: int) -> nn.GroupNorm:
    num_groups = min(32, ch)
    while ch % num_groups != 0 and num_groups > 1:
        num_groups -= 1
    return nn.GroupNorm(num_groups=num_groups, num_channels=ch)


class ResBlock(nn.Module):
    """Residual block with adaptive group norm time conditioning."""

    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int) -> None:
        super().__init__()
        self.norm1 = _norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = _norm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch * 2),
        )
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(
        self, x: torch.Tensor, t_emb: torch.Tensor, adapter: Optional[nn.Module] = None
    ) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = self.norm2(h)
        scale, shift = self.time_mlp(t_emb).chunk(2, dim=1)
        h = h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = F.silu(h)
        h = self.conv2(h)
        if adapter is not None:
            h = h + adapter(h)
        return h + self.shortcut(x)


class SelfAttention2D(nn.Module):
    """Self-attention block operating on spatial feature maps."""

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.norm = _norm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q = self.norm(x).view(b, c, h * w).permute(0, 2, 1)
        out, _ = self.attn(q, q, q)
        out = out.permute(0, 2, 1).view(b, c, h, w)
        return x + out


class UNet2D(nn.Module):
    """Compact UNet for latent-space velocity estimation.

    Args:
        in_channels: number of input channels (latent channels, e.g. 4).
        out_channels: number of output channels (same as input for RF).
        model_channels: base channel width.
        channel_mult: channel multipliers per downsampling level.
        num_res_blocks: residual blocks per level.
        time_emb_dim: dimension of the sinusoidal time embedding.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        model_channels: int = 64,
        channel_mult: Sequence[int] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        time_emb_dim: int = 256,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim
        self.num_res_blocks = num_res_blocks
        self.channel_mult = tuple(channel_mult)

        self.time_embed = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )
        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        # Encoder
        self.encoder = nn.ModuleList()
        ch = model_channels
        self.skip_channels: List[int] = []
        for i, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(ch, out_ch, time_emb_dim))
                ch = out_ch
            self.skip_channels.append(ch)
            down = (
                nn.Conv2d(ch, ch, 3, stride=2, padding=1)
                if i != len(channel_mult) - 1
                else nn.Identity()
            )
            self.encoder.append(nn.ModuleList([blocks, down]))

        # Middle
        self.middle = nn.ModuleList(
            [
                ResBlock(ch, ch, time_emb_dim),
                SelfAttention2D(ch),
                ResBlock(ch, ch, time_emb_dim),
            ]
        )

        # Decoder
        self.decoder = nn.ModuleList()
        for i, mult in enumerate(reversed(channel_mult)):
            out_ch = model_channels * mult
            skip_ch = self.skip_channels.pop()
            blocks = nn.ModuleList()
            blocks.append(ResBlock(ch + skip_ch, out_ch, time_emb_dim))
            ch = out_ch
            for _ in range(num_res_blocks - 1):
                blocks.append(ResBlock(ch, out_ch, time_emb_dim))
            up = (
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(ch, ch, 3, padding=1),
                )
                if i != len(channel_mult) - 1
                else nn.Identity()
            )
            self.decoder.append(nn.ModuleList([blocks, up]))

        self.output_conv = nn.Sequential(
            _norm(ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1),
        )

    @property
    def num_adapter_positions(self) -> int:
        """Total number of ResBlocks where an adapter can be inserted."""
        n_levels = len(self.channel_mult)
        return n_levels * self.num_res_blocks + 2 + n_levels * self.num_res_blocks

    @property
    def adapter_channels(self) -> List[int]:
        """Output channel count of each ResBlock in forward order."""
        chs = []
        for blocks, _ in self.encoder:
            for block in blocks:
                chs.append(block.conv2.out_channels)
        for layer in self.middle:
            if isinstance(layer, ResBlock):
                chs.append(layer.conv2.out_channels)
        for blocks, _ in self.decoder:
            for block in blocks:
                chs.append(block.conv2.out_channels)
        return chs

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        adapters: Optional[Sequence[Optional[nn.Module]]] = None,
    ) -> torch.Tensor:
        """Args:
            x: (B, C, H, W) latent tensor.
            t: (B,) continuous time in [0, 1] (or integer diffusion steps).
            adapters: optional list of adapter modules, one per ResBlock.
        """
        t_emb = get_timestep_embedding(t, self.time_emb_dim)
        t_emb = self.time_embed(t_emb)

        h = self.input_conv(x)
        adapter_idx = 0
        skips: List[torch.Tensor] = []

        for blocks, down in self.encoder:
            for block in blocks:
                adapter = adapters[adapter_idx] if adapters else None
                h = block(h, t_emb, adapter)
                adapter_idx += 1
            skips.append(h)
            h = down(h)

        for layer in self.middle:
            if isinstance(layer, ResBlock):
                adapter = adapters[adapter_idx] if adapters else None
                h = layer(h, t_emb, adapter)
                adapter_idx += 1
            else:
                h = layer(h)

        for blocks, up in self.decoder:
            h = torch.cat([h, skips.pop()], dim=1)
            for block in blocks:
                adapter = adapters[adapter_idx] if adapters else None
                h = block(h, t_emb, adapter)
                adapter_idx += 1
            h = up(h)

        return self.output_conv(h)

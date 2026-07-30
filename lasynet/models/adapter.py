"""Parameter-efficient bottleneck adapters (AdaptFormer-style)."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBottleneckAdapter(nn.Module):
    """Lightweight convolutional adapter with a residual path.

    The adapter is initialized to be close to zero so that the frozen
    backbone is not perturbed at the beginning of training.
    """

    def __init__(self, channels: int, bottleneck_channels: Optional[int] = None) -> None:
        super().__init__()
        bottleneck_channels = bottleneck_channels or max(channels // 4, 16)
        self.down = nn.Conv2d(channels, bottleneck_channels, 1)
        self.act = nn.GELU()
        self.up = nn.Conv2d(bottleneck_channels, channels, 1)
        # Zero-init the output conv so the adapter starts as identity.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.down(x)
        h = self.act(h)
        h = self.up(h)
        return x + self.scale * h


def make_adapter_list(channels: int, num_positions: int) -> nn.ModuleList:
    """Create a trainable adapter for every insertion position."""
    return nn.ModuleList([ConvBottleneckAdapter(channels) for _ in range(num_positions)])

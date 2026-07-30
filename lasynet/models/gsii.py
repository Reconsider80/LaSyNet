"""Gated Symbiotic Information Interaction (G-SII) module."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(ch: int) -> nn.GroupNorm:
    num_groups = min(32, ch)
    while ch % num_groups != 0 and num_groups > 1:
        num_groups -= 1
    return nn.GroupNorm(num_groups=num_groups, num_channels=ch)


class GSII(nn.Module):
    """Timestep-aware bidirectional routing between enhancement and segmentation.

    Given latent representations at time t, the module computes two gating
    scalars g_{s->e}(t) and g_{e->s}(t) and uses cross-attention to exchange
    structural priors and texture details between the two tasks.

    Reference: Eq. (3)-(7) of the paper.
    """

    def __init__(
        self,
        latent_dim: int,
        time_emb_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.time_emb_dim = time_emb_dim

        self.gate_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, 2),
        )

        self.norm_enh = _norm(latent_dim)
        self.norm_seg = _norm(latent_dim)
        self.attn_enh = nn.MultiheadAttention(
            latent_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.attn_seg = nn.MultiheadAttention(
            latent_dim, num_heads, dropout=dropout, batch_first=True
        )

    def _norm_seq(self, x: torch.Tensor, norm: nn.GroupNorm) -> torch.Tensor:
        """Apply GroupNorm to a sequence tensor (B, L, C)."""
        # x: (B, L, C) -> (B, C, L) -> norm -> (B, L, C)
        return norm(x.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()

    def forward(
        self,
        z_enh: torch.Tensor,
        z_seg: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_enh: (B, latent_dim, H, W) enhancement latent.
            z_seg: (B, latent_dim, H, W) segmentation latent.
            t_emb: (B, time_emb_dim) time embedding.

        Returns:
            (F_oi, F_os): symbiotic guided features of the same shape.
        """
        b, c, h, w = z_enh.shape
        gates = torch.sigmoid(self.gate_mlp(t_emb))  # (B, 2)
        g_s2e = gates[:, 0].view(b, 1, 1)
        g_e2s = gates[:, 1].view(b, 1, 1)

        x_e = z_enh.view(b, c, h * w).permute(0, 2, 1).contiguous()  # (B, HW, C)
        x_s = z_seg.view(b, c, h * w).permute(0, 2, 1).contiguous()

        # Segmentation -> Enhancement
        q_e = self._norm_seq(x_e, self.norm_enh)
        k_s = self._norm_seq(x_s, self.norm_seg)
        v_s = k_s
        attn_e, _ = self.attn_enh(q_e, k_s, v_s)
        F_oic = g_s2e * attn_e
        F_oi = F.relu(F_oic) + x_e

        # Enhancement -> Segmentation
        q_s = self._norm_seq(x_s, self.norm_seg)
        k_e = self._norm_seq(x_e, self.norm_enh)
        v_e = k_e
        attn_s, _ = self.attn_seg(q_s, k_e, v_e)
        F_osc = g_e2s * attn_s
        F_os = F.relu(F_osc) + x_s

        F_oi = F_oi.permute(0, 2, 1).view(b, c, h, w)
        F_os = F_os.permute(0, 2, 1).view(b, c, h, w)
        return F_oi, F_os

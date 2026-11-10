"""Upgraded Gated Symbiotic Information Interaction (G-SII) module.

CVPR revision relative to the AAAI draft:
  - Frequency-aware gating conditioned on Embed(t) and F_fft(Z_enh, Z_seg)
  - Token Discrepancy Extractor (TD-X) for agreement / difference / residual cues
  - Dual-domain (spectral + spatial) bidirectional routing
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


class FrequencyStatsExtractor(nn.Module):
    """Extract global frequency-domain statistics F_fft(Z_enh, Z_seg).

    Computes rFFT magnitudes of both latents, pools them spatially, and
    projects the concatenated spectrum into a compact descriptor used by the
    time-frequency-aware gating MLP (Eq. 6).
    """

    def __init__(self, latent_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(latent_dim * 4, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def _spectral_pool(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, C, H, W) -> global mean/std of FFT magnitude
        spec = torch.fft.rfft2(z.float(), norm="ortho")
        mag = torch.abs(spec)
        mean = mag.mean(dim=(-2, -1))
        std = mag.std(dim=(-2, -1), unbiased=False)
        return torch.cat([mean, std], dim=-1)

    def forward(self, z_enh: torch.Tensor, z_seg: torch.Tensor) -> torch.Tensor:
        feats = torch.cat(
            [self._spectral_pool(z_enh), self._spectral_pool(z_seg)], dim=-1
        )
        return self.proj(feats.to(dtype=z_enh.dtype))


class TokenDiscrepancyExtractor(nn.Module):
    """TD-X: agreement / difference / residual discrepancy cues (from MedFlowSeg).

    Given query tokens T_q and context tokens T_c:
        T_a = T_q ⊙ T_c
        T_d = |T_q - T_c|
        T_r = T_q - T_c
    The fused evidence token recalibrates both streams before cross-attention.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.enc_a = nn.Sequential(nn.Linear(dim, dim), nn.SiLU())
        self.enc_d = nn.Sequential(nn.Linear(dim, dim), nn.SiLU())
        self.enc_r = nn.Sequential(nn.Linear(dim, dim), nn.SiLU())
        self.fuse = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.recal_q = nn.Linear(dim * 2, dim)
        self.recal_c = nn.Linear(dim * 2, dim)

    def forward(
        self, t_q: torch.Tensor, t_c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t_a = t_q * t_c
        t_d = (t_q - t_c).abs()
        t_r = t_q - t_c
        evidence = self.fuse(
            torch.cat([self.enc_a(t_a), self.enc_d(t_d), self.enc_r(t_r)], dim=-1)
        )
        t_q_hat = self.recal_q(torch.cat([t_q, evidence], dim=-1))
        t_c_hat = self.recal_c(torch.cat([t_c, evidence], dim=-1))
        return t_q_hat, t_c_hat


class GSII(nn.Module):
    """Frequency-aware, discrepancy-guided bidirectional G-SII routing.

    Reference: CVPR revision Eqs. (6)-(10).
    """

    def __init__(
        self,
        latent_dim: int,
        time_emb_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        freq_emb_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.time_emb_dim = time_emb_dim
        self.freq_emb_dim = freq_emb_dim or time_emb_dim

        self.freq_stats = FrequencyStatsExtractor(latent_dim, self.freq_emb_dim)
        self.gate_mlp = nn.Sequential(
            nn.Linear(time_emb_dim + self.freq_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, 2),
        )

        self.norm_enh = _norm(latent_dim)
        self.norm_seg = _norm(latent_dim)
        self.tdx_enh = TokenDiscrepancyExtractor(latent_dim)
        self.tdx_seg = TokenDiscrepancyExtractor(latent_dim)
        self.attn_enh = nn.MultiheadAttention(
            latent_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.attn_seg = nn.MultiheadAttention(
            latent_dim, num_heads, dropout=dropout, batch_first=True
        )

    def _norm_seq(self, x: torch.Tensor, norm: nn.GroupNorm) -> torch.Tensor:
        """Apply GroupNorm to a sequence tensor (B, L, C)."""
        return norm(x.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()

    def forward(
        self,
        z_enh: torch.Tensor,
        z_seg: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_enh: (B, latent_dim, H, W) enhancement latent at timestep t.
            z_seg: (B, latent_dim, H, W) segmentation latent at timestep t.
            t_emb: (B, time_emb_dim) time embedding.

        Returns:
            (F_oi, F_os): symbiotic guided features of the same spatial shape.
        """
        b, c, h, w = z_enh.shape

        # Eq. (6): time-frequency-aware gating scalars.
        f_fft = self.freq_stats(z_enh, z_seg)
        gates = torch.sigmoid(self.gate_mlp(torch.cat([t_emb, f_fft], dim=-1)))
        g_s2e = gates[:, 0].view(b, 1, 1)
        g_e2s = gates[:, 1].view(b, 1, 1)

        x_e = z_enh.view(b, c, h * w).permute(0, 2, 1).contiguous()
        x_s = z_seg.view(b, c, h * w).permute(0, 2, 1).contiguous()

        # Segmentation -> Enhancement (Eq. 7-8) with TD-X discrepancy cues.
        q_e = self._norm_seq(x_e, self.norm_enh)
        k_s = self._norm_seq(x_s, self.norm_seg)
        q_e, k_s = self.tdx_enh(q_e, k_s)
        attn_e, _ = self.attn_enh(q_e, k_s, k_s)
        F_oic = g_s2e * attn_e
        F_oi = F.relu(F_oic) + x_e

        # Enhancement -> Segmentation (Eq. 9-10) with TD-X discrepancy cues.
        q_s = self._norm_seq(x_s, self.norm_seg)
        k_e = self._norm_seq(x_e, self.norm_enh)
        q_s, k_e = self.tdx_seg(q_s, k_e)
        attn_s, _ = self.attn_seg(q_s, k_e, k_e)
        F_osc = g_e2s * attn_s
        F_os = F.relu(F_osc) + x_s

        F_oi = F_oi.permute(0, 2, 1).view(b, c, h, w)
        F_os = F_os.permute(0, 2, 1).view(b, c, h, w)
        return F_oi, F_os

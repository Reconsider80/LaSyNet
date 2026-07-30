"""LaSyNet: Latent Symbiosis Network for joint medical image enhancement and segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adapter import ConvBottleneckAdapter
from .gsii import GSII
from .seg_encoder import MaskEncoder, SegDecoder, SegEncoder
from .unet import UNet2D, get_timestep_embedding
from .vae import Autoencoder


class LaSyNet(nn.Module):
    """Unified Rectified-Flow framework with adapter-guided G-SII routing.

    Args:
        image_channels: input image channels (usually 1 for medical images).
        num_classes: number of segmentation classes.  Use 1 for binary masks.
        latent_dim: latent channel dimension (4 in LDM-style VAEs).
        seg_channels: channel dimension of the spatial priors Flq.
        time_emb_dim: time embedding dimension.
        vae_*: configuration of the first-stage autoencoder.
        unet_*: configuration of the frozen Rectified-Flow UNet backbone.
        gsii_num_heads: number of attention heads in the G-SII module.
        beta: weight of the segmentation RF loss.
        lambda_seg: weight of the pixel-space segmentation loss.
        freeze_backbone: whether the UNet backbone is frozen (PEFT setup).
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.image_channels = cfg["image_channels"]
        self.num_classes = cfg["num_classes"]
        self.latent_dim = cfg["latent_dim"]
        self.seg_channels = cfg["seg_channels"]
        self.time_emb_dim = cfg["time_emb_dim"]
        self.beta = cfg["beta"]
        self.lambda_seg = cfg["lambda_seg"]

        self.vae = Autoencoder(
            in_channels=self.image_channels,
            latent_channels=self.latent_dim,
            base_channels=cfg["vae_base_channels"],
            channel_mult=tuple(cfg["vae_channel_mult"]),
            num_res_blocks=cfg["vae_num_res_blocks"],
            kl_weight=cfg["vae_kl_weight"],
        )
        self.seg_encoder = SegEncoder(
            in_channels=self.image_channels,
            out_channels=self.seg_channels,
            base_channels=cfg["seg_base_channels"],
            downsample_steps=cfg["seg_downsample_steps"],
        )
        self.mask_encoder = MaskEncoder(
            in_channels=self.num_classes if self.num_classes > 1 else 1,
            latent_channels=self.latent_dim,
            base_channels=cfg["vae_base_channels"],
            downsample_steps=cfg["seg_downsample_steps"],
        )
        self.seg_decoder = SegDecoder(
            in_channels=self.latent_dim,
            out_channels=self.num_classes,
            base_channels=cfg["unet_model_channels"],
            upsample_steps=len(cfg["vae_channel_mult"]) - 1,
        )

        self.backbone = UNet2D(
            in_channels=self.latent_dim,
            out_channels=self.latent_dim,
            model_channels=cfg["unet_model_channels"],
            channel_mult=tuple(cfg["unet_channel_mult"]),
            num_res_blocks=cfg["unet_num_res_blocks"],
            time_emb_dim=self.time_emb_dim,
        )
        if cfg.get("freeze_backbone", True):
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Per-block adapters inserted into the frozen backbone.
        adapter_channels = self.backbone.adapter_channels
        self.enh_adapters = nn.ModuleList(
            [ConvBottleneckAdapter(ch) for ch in adapter_channels]
        )
        self.seg_adapters = nn.ModuleList(
            [ConvBottleneckAdapter(ch) for ch in adapter_channels]
        )

        # Project concatenated conditions into the backbone's latent channel count.
        enh_in_ch = self.latent_dim * 3  # zt, z_lq, F_oi
        seg_in_ch = self.latent_dim * 2 + self.seg_channels  # zt, Flq, F_os
        self.enh_input_proj = nn.Conv2d(enh_in_ch, self.latent_dim, 3, padding=1)
        self.seg_input_proj = nn.Conv2d(seg_in_ch, self.latent_dim, 3, padding=1)

        self.gsii = GSII(
            self.latent_dim, self.time_emb_dim, num_heads=cfg["gsii_num_heads"]
        )

        self._eps = 1e-7

    def _match_resolution(self, cond: torch.Tensor, target_size: tuple) -> torch.Tensor:
        """Resize a spatial condition to the target latent resolution if needed."""
        if cond.shape[2:] == target_size:
            return cond
        return F.interpolate(cond, size=target_size, mode="bilinear", align_corners=False)

    def encode_targets(
        self, x_lq: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode inputs and targets into the latent space."""
        z_lq = self.vae.get_latent(x_lq)
        z_enh_1 = self.vae.get_latent(gt)

        if self.num_classes > 1:
            mask_oh = (
                F.one_hot(mask.long(), num_classes=self.num_classes)
                .permute(0, 3, 1, 2)
                .float()
            )
            z_seg_1 = self.mask_encoder(mask_oh)
        else:
            z_seg_1 = self.mask_encoder(mask.float())
        return z_lq, z_enh_1, z_seg_1

    def _predict_velocity(
        self,
        zt: torch.Tensor,
        cond_a: torch.Tensor,
        cond_b: torch.Tensor,
        t: torch.Tensor,
        input_proj: nn.Module,
        adapters: nn.ModuleList,
    ) -> torch.Tensor:
        """Run one branch of the adapter-guided Rectified-Flow backbone."""
        x = torch.cat([zt, cond_a, cond_b], dim=1)
        x = input_proj(x)
        return self.backbone(x, t, adapters)

    def forward(self, x_lq: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> dict:
        """Joint training forward pass."""
        z_lq, z_enh_1, z_seg_1 = self.encode_targets(x_lq, gt, mask)
        b = z_enh_1.shape[0]
        device = z_enh_1.device

        # Sample timestep and noise for the Rectified Flow straight-line trajectory.
        t = torch.rand(b, device=device)
        z0_enh = torch.randn_like(z_enh_1)
        z0_seg = torch.randn_like(z_seg_1)
        t4 = t.view(b, 1, 1, 1)
        zt_enh = (1.0 - t4) * z0_enh + t4 * z_enh_1
        zt_seg = (1.0 - t4) * z0_seg + t4 * z_seg_1

        # Spatial priors and G-SII routing.
        F_lq = self.seg_encoder(x_lq)
        F_lq = self._match_resolution(F_lq, z_lq.shape[2:])
        t_emb = get_timestep_embedding(t, self.time_emb_dim)
        F_oi, F_os = self.gsii(zt_enh, zt_seg, t_emb)

        # Predict the velocity fields for both tasks.
        v_enh = self._predict_velocity(
            zt_enh, z_lq, F_oi, t, self.enh_input_proj, self.enh_adapters
        )
        v_seg = self._predict_velocity(
            zt_seg, F_lq, F_os, t, self.seg_input_proj, self.seg_adapters
        )

        # Rectified-Flow losses (Eq. 8, 9).
        target_enh = z_enh_1 - z0_enh
        target_seg = z_seg_1 - z0_seg
        loss_rf_enh = F.mse_loss(v_enh, target_enh)
        loss_rf_seg = F.mse_loss(v_seg, target_seg)

        # Approximate clean latent from the predicted velocity for output supervision.
        z1_enh_pred = zt_enh + (1.0 - t4) * v_enh
        z1_seg_pred = zt_seg + (1.0 - t4) * v_seg
        I_p = self.vae.decode(z1_enh_pred)
        S_logits = self.seg_decoder(z1_seg_pred)

        # Pixel-space losses (Eq. 10, 11).
        loss_enh = F.mse_loss(I_p, gt)
        if self.num_classes == 1:
            loss_seg = self._binary_segmentation_loss(S_logits, mask)
            S_p = torch.sigmoid(S_logits)
        else:
            loss_seg = self._multiclass_segmentation_loss(S_logits, mask)
            S_p = torch.softmax(S_logits, dim=1)

        loss = (
            loss_rf_enh
            + self.beta * loss_rf_seg
            + loss_enh
            + self.lambda_seg * loss_seg
        )

        return {
            "loss": loss,
            "loss_rf_enh": loss_rf_enh.detach(),
            "loss_rf_seg": loss_rf_seg.detach(),
            "loss_enh": loss_enh.detach(),
            "loss_seg": loss_seg.detach(),
            "I_p": I_p.detach(),
            "S_p": S_p.detach(),
        }

    def _binary_segmentation_loss(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Weighted binary cross-entropy favouring the foreground class."""
        target = target.float()
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        pos_ratio = target.mean().clamp_min(self._eps)
        pos_weight = (1.0 - pos_ratio) / pos_ratio
        weights = torch.where(target > 0.5, pos_weight, 1.0)
        return (bce * weights).mean()

    def _multiclass_segmentation_loss(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Cross-entropy with median-frequency class weighting."""
        b, num_classes, h, w = logits.shape
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, num_classes)
        target_flat = target.long().view(-1)
        counts = torch.bincount(target_flat, minlength=num_classes).float()
        weights = counts.max() / (counts + self._eps)
        return F.cross_entropy(logits_flat, target_flat, weight=weights.to(logits.device))

    @torch.no_grad()
    def sample(
        self,
        x_lq: torch.Tensor,
        num_steps: int = 25,
        return_latents: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Deterministic Euler ODE integration for inference (Eq. 1-2, 3-7)."""
        b = x_lq.shape[0]
        device = x_lq.device
        z_lq = self.vae.get_latent(x_lq)
        F_lq = self._match_resolution(self.seg_encoder(x_lq), z_lq.shape[2:])
        latent_h, latent_w = z_lq.shape[2:]

        z_enh = torch.randn(b, self.latent_dim, latent_h, latent_w, device=device)
        z_seg = torch.randn(b, self.latent_dim, latent_h, latent_w, device=device)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((b,), i * dt, device=device)
            t_emb = get_timestep_embedding(t, self.time_emb_dim)
            F_oi, F_os = self.gsii(z_enh, z_seg, t_emb)
            v_enh = self._predict_velocity(
                z_enh, z_lq, F_oi, t, self.enh_input_proj, self.enh_adapters
            )
            v_seg = self._predict_velocity(
                z_seg, F_lq, F_os, t, self.seg_input_proj, self.seg_adapters
            )
            z_enh = z_enh + dt * v_enh
            z_seg = z_seg + dt * v_seg

        I_p = self.vae.decode(z_enh)
        S_logits = self.seg_decoder(z_seg)
        S_p = torch.sigmoid(S_logits) if self.num_classes == 1 else torch.softmax(S_logits, dim=1)
        if return_latents:
            return I_p, S_p, z_enh, z_seg
        return I_p, S_p

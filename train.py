"""Training script for LaSyNet."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lasynet.data import get_dataset
from lasynet.models import LaSyNet
from lasynet.utils import AverageMeter, load_config, save_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LaSyNet")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    return parser.parse_args()


def build_optimizer(model: torch.nn.Module, cfg: dict) -> torch.optim.Optimizer:
    trainable = [p for p in model.parameters() if p.requires_grad]
    lr = cfg.get("lr", 1e-4)
    weight_decay = cfg.get("weight_decay", 1e-2)
    return torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict) -> torch.optim.lr_scheduler.LRScheduler:
    name = cfg.get("scheduler", "none").lower()
    if name == "cosine":
        epochs = cfg.get("epochs", 100)
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=cfg.get("lr_min", 1e-6)
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.get("step_size", 30), gamma=cfg.get("gamma", 0.5)
        )
    return torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=1.0, total_iters=1)


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter | None,
    use_amp: bool,
) -> dict:
    model.train()
    losses = {
        "total": AverageMeter(),
        "rf_enh": AverageMeter(),
        "rf_seg": AverageMeter(),
        "enh": AverageMeter(),
        "seg": AverageMeter(),
    }
    start = time.time()

    for step, (x_lq, gt, mask) in enumerate(dataloader):
        x_lq = x_lq.to(device)
        gt = gt.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()
        with autocast(enabled=use_amp):
            out = model(x_lq, gt, mask)
            loss = out["loss"]

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        n = x_lq.size(0)
        losses["total"].update(out["loss"].item(), n)
        losses["rf_enh"].update(out["loss_rf_enh"].item(), n)
        losses["rf_seg"].update(out["loss_rf_seg"].item(), n)
        losses["enh"].update(out["loss_enh"].item(), n)
        losses["seg"].update(out["loss_seg"].item(), n)

        if step % 10 == 0:
            print(
                f"Epoch [{epoch}] step [{step}/{len(dataloader)}] "
                f"loss={losses['total'].avg:.4f} "
                f"rf_enh={losses['rf_enh'].avg:.4f} "
                f"rf_seg={losses['rf_seg'].avg:.4f} "
                f"enh={losses['enh'].avg:.4f} seg={losses['seg'].avg:.4f}"
            )
        if writer is not None:
            global_step = epoch * len(dataloader) + step
            writer.add_scalar("train/loss", out["loss"].item(), global_step)
            writer.add_scalar("train/rf_enh", out["loss_rf_enh"].item(), global_step)
            writer.add_scalar("train/rf_seg", out["loss_rf_seg"].item(), global_step)
            writer.add_scalar("train/enh", out["loss_enh"].item(), global_step)
            writer.add_scalar("train/seg", out["loss_seg"].item(), global_step)

    elapsed = time.time() - start
    print(f"Epoch {epoch} finished in {elapsed:.1f}s")
    return {k: v.avg for k, v in losses.items()}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output_dir / "config.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = LaSyNet(cfg).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {trainable:,} trainable / {total:,} total")

    train_dataset = get_dataset(cfg, split="train")
    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.get("batch_size", 4),
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    scaler = GradScaler(enabled=cfg.get("amp", False))
    writer = SummaryWriter(log_dir=output_dir / "logs") if cfg.get("use_tensorboard", True) else None

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    epochs = cfg.get("epochs", 100)
    for epoch in range(start_epoch, epochs):
        train_one_epoch(
            model, dataloader, optimizer, scaler, device, epoch, writer, cfg.get("amp", False)
        )
        scheduler.step()

        if (epoch + 1) % cfg.get("save_every", 10) == 0:
            ckpt_path = output_dir / f"checkpoint_epoch_{epoch+1:03d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "cfg": cfg,
                },
                ckpt_path,
            )
            print(f"Saved checkpoint to {ckpt_path}")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()

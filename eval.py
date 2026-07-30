"""Evaluation script for LaSyNet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from lasynet.data import get_dataset
from lasynet.models import LaSyNet
from lasynet.utils import AverageMeter, compute_dice, compute_iou, compute_psnr, compute_ssim, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LaSyNet")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save predictions")
    parser.add_argument("--num_steps", type=int, default=None, help="ODE sampling steps (override config)")
    parser.add_argument("--save_images", action="store_true", help="Save sample images")
    return parser.parse_args()


def save_image(path: Path, tensor: torch.Tensor) -> None:
    from PIL import Image

    arr = (tensor.squeeze().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def save_mask(path: Path, tensor: torch.Tensor, num_classes: int) -> None:
    from PIL import Image

    if num_classes == 1:
        arr = (tensor.squeeze().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    else:
        arr = tensor.argmax(dim=0).cpu().numpy().astype(np.uint8)
    Image.fromarray(arr).save(path)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = LaSyNet(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    num_steps = args.num_steps if args.num_steps is not None else cfg.get("num_steps", 25)
    num_classes = cfg.get("num_classes", 1)
    dataset = get_dataset(cfg, split="test")
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.get("batch_size", 1),
        shuffle=False,
        num_workers=cfg.get("num_workers", 2),
        pin_memory=False,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_images:
        (output_dir / "images").mkdir(exist_ok=True)
        (output_dir / "masks").mkdir(exist_ok=True)

    psnr_lq = AverageMeter()
    psnr_out = AverageMeter()
    ssim_lq = AverageMeter()
    ssim_out = AverageMeter()
    dice_meter = AverageMeter()
    iou_meter = AverageMeter()

    with torch.no_grad():
        for idx, (x_lq, gt, mask) in enumerate(dataloader):
            x_lq = x_lq.to(device)
            gt = gt.to(device)
            mask = mask.to(device)
            I_p, S_p = model.sample(x_lq, num_steps=num_steps)

            # Move to CPU/numpy for metrics.
            x_lq_np = x_lq.cpu().numpy()
            gt_np = gt.cpu().numpy()
            I_p_np = I_p.cpu().numpy()
            mask_np = mask.cpu().numpy()
            S_p_np = S_p.cpu().numpy()

            b = x_lq_np.shape[0]
            for i in range(b):
                psnr_lq.update(compute_psnr(x_lq_np[i, 0], gt_np[i, 0]), 1)
                psnr_out.update(compute_psnr(I_p_np[i, 0], gt_np[i, 0]), 1)
                ssim_lq.update(compute_ssim(x_lq_np[i, 0], gt_np[i, 0]), 1)
                ssim_out.update(compute_ssim(I_p_np[i, 0], gt_np[i, 0]), 1)

                if num_classes == 1:
                    pred_mask = (S_p_np[i, 0] > 0.5).astype(np.uint8)
                    target_mask = mask_np[i].astype(np.uint8)
                else:
                    pred_mask = S_p_np[i].argmax(axis=0).astype(np.uint8)
                    target_mask = mask_np[i].astype(np.uint8)
                dice_meter.update(compute_dice(pred_mask, target_mask, num_classes)["dice"], 1)
                iou_meter.update(compute_iou(pred_mask, target_mask, num_classes)["miou"], 1)

                if args.save_images and idx < 5:
                    save_image(output_dir / "images" / f"{idx}_{i}_lq.png", x_lq[i])
                    save_image(output_dir / "images" / f"{idx}_{i}_out.png", I_p[i])
                    save_image(output_dir / "images" / f"{idx}_{i}_gt.png", gt[i])
                    save_mask(output_dir / "masks" / f"{idx}_{i}_pred.png", S_p[i], num_classes)
                    save_mask(output_dir / "masks" / f"{idx}_{i}_gt.png", mask[i], num_classes)

    results = {
        "psnr_input": psnr_lq.avg,
        "psnr_output": psnr_out.avg,
        "ssim_input": ssim_lq.avg,
        "ssim_output": ssim_out.avg,
        "dice": dice_meter.avg,
        "miou": iou_meter.avg,
        "num_steps": num_steps,
    }
    print(json.dumps(results, indent=2))
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

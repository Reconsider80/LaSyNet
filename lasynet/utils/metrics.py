"""Evaluation metrics: PSNR, SSIM, Dice, and mIoU."""

from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compute_psnr(img1: np.ndarray, img2: np.ndarray, data_range: float = 1.0) -> float:
    """Compute PSNR between two images in [0, data_range]."""
    return float(peak_signal_noise_ratio(img1, img2, data_range=data_range))


def compute_ssim(img1: np.ndarray, img2: np.ndarray, data_range: float = 1.0) -> float:
    """Compute SSIM between two images."""
    return float(structural_similarity(img1, img2, data_range=data_range))


def _one_hot(mask: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert a (H, W) class-index mask to (num_classes, H, W) one-hot."""
    h, w = mask.shape
    oh = np.zeros((num_classes, h, w), dtype=np.float32)
    for c in range(num_classes):
        oh[c] = (mask == c).astype(np.float32)
    return oh


def compute_dice(pred: np.ndarray, target: np.ndarray, num_classes: int = 1) -> dict:
    """Compute Dice coefficient per class and averaged over classes present in target."""
    if num_classes == 1:
        pred = pred.astype(np.float32)
        target = target.astype(np.float32)
        inter = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = (2.0 * inter + 1e-7) / (union + 1e-7)
        return {"dice": float(dice)}

    pred_oh = _one_hot(pred, num_classes)
    target_oh = _one_hot(target, num_classes)
    per_class = []
    for c in range(num_classes):
        inter = (pred_oh[c] * target_oh[c]).sum()
        union = pred_oh[c].sum() + target_oh[c].sum()
        per_class.append((2.0 * inter + 1e-7) / (union + 1e-7))
    per_class = np.array(per_class)
    present = target_oh.sum(axis=(1, 2)) > 0
    avg = per_class[present].mean() if present.any() else 0.0
    return {"dice": float(avg), "per_class_dice": per_class.tolist()}


def compute_iou(pred: np.ndarray, target: np.ndarray, num_classes: int = 1) -> dict:
    """Compute mean IoU per class and averaged over classes present in target."""
    if num_classes == 1:
        pred = pred.astype(np.float32)
        target = target.astype(np.float32)
        inter = (pred * target).sum()
        union = pred.sum() + target.sum() - inter
        iou = (inter + 1e-7) / (union + 1e-7)
        return {"miou": float(iou)}

    pred_oh = _one_hot(pred, num_classes)
    target_oh = _one_hot(target, num_classes)
    per_class = []
    for c in range(num_classes):
        inter = (pred_oh[c] * target_oh[c]).sum()
        union = pred_oh[c].sum() + target_oh[c].sum() - inter
        per_class.append((inter + 1e-7) / (union + 1e-7))
    per_class = np.array(per_class)
    present = target_oh.sum(axis=(1, 2)) > 0
    avg = per_class[present].mean() if present.any() else 0.0
    return {"miou": float(avg), "per_class_iou": per_class.tolist()}


class AverageMeter:
    """Simple running average accumulator."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / (self.count + 1e-8)

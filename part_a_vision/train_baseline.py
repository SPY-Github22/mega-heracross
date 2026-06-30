#!/usr/bin/env python3
"""
Phase 7 — Task 3 & 4: DeepLabV3+ Baseline Training Loop
========================================================
Trains a DeepLabV3+ (ResNet-50 backbone, 6 input channels) on synthetic
Koramangala fused optical-SAR tiles.

Key design decisions:
  - BCEWithLogitsLoss with pos_weight=6.0 (roads ~15% of pixels)
  - AdamW optimizer, lr=1e-4, weight_decay=1e-4
  - CosineAnnealingLR scheduler (50 epochs)
  - Best model saved by validation IoU
  - CSV logging of all metrics for TensorBoard / analysis
  - Baseline IoU target: >0.30 on synthetic val split

Place at: part_a_vision/train_baseline.py

Usage:
  python train_baseline.py --batch-size 4 --epochs 50
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("train_baseline")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(_h)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_BATCH_SIZE: int = 4
DEFAULT_EPOCHS: int = 50
DEFAULT_LR: float = 1e-4
DEFAULT_WEIGHT_DECAY: float = 1e-4
POS_WEIGHT_ROAD: float = 6.0
TARGET_IOU: float = 0.30
NUM_TILES: int = 200
TILE_SIZE: int = 512
INPUT_CHANNELS: int = 6
NUM_CLASSES: int = 1  # binary road segmentation

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    pred_logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute IoU, Dice, and pixel accuracy from raw logits.

    Parameters
    ----------
    pred_logits : (B, 1, H, W) or (B, H, W)
        Raw logits from model output.
    targets : (B, H, W)
        Binary ground truth {0, 1}.
    threshold : float
        Decision threshold for sigmoid output.

    Returns
    -------
    dict with keys: iou, dice, pixel_acc, precision, recall
    """
    if pred_logits.dim() == 4 and pred_logits.shape[1] == 1:
        pred_logits = pred_logits.squeeze(1)

    probs = torch.sigmoid(pred_logits)
    preds = (probs > threshold).long()

    targets = targets.long()

    intersection = (preds * targets).sum().float()
    union = (preds + targets).clamp(0, 1).sum().float()
    iou = (intersection / (union + 1e-8)).item()

    dice = (2 * intersection / (preds.sum().float() + targets.sum().float() + 1e-8)).item()

    correct = (preds == targets).sum().float()
    total = targets.numel()
    pixel_acc = (correct / total).item()

    # Precision / Recall
    tp = intersection
    fp = (preds * (1 - targets)).sum().float()
    fn = ((1 - preds) * targets).sum().float()
    precision = (tp / (tp + fp + 1e-8)).item()
    recall = (tp / (tp + fn + 1e-8)).item()

    return {
        "iou": iou,
        "dice": dice,
        "pixel_acc": pixel_acc,
        "precision": precision,
        "recall": recall,
    }


def metrics_avg(metrics_list: list) -> Dict[str, float]:
    """Average a list of metric dicts."""
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    return {k: np.mean([m[k] for m in metrics_list]) for k in keys}


# ---------------------------------------------------------------------------
# Model Builder
# ---------------------------------------------------------------------------

def build_model(
    input_channels: int = INPUT_CHANNELS,
    num_classes: int = NUM_CLASSES,
    backbone: str = "resnet50",
    pretrained: bool = True,
) -> nn.Module:
    """
    Build DeepLabV3+ model with custom input channels.

    Since torchvision's DeepLabV3 expects 3-channel RGB, we:
      1. Create the standard model
      2. Replace conv1 with a 6-channel variant (copy-paste RGB weights,
         init extra channels with Gaussian noise scaled to match)
      3. DeepLabV3+ = DeepLabV3 + decoder (use deeplabv3_resnet50)

    For true DeepLabV3+ we use segmentation-models-pytorch if available,
    otherwise fall back to torchvision DeepLabV3.
    """
    try:
        import segmentation_models_pytorch as smp

        logger.info("Using segmentation-models-pytorch for DeepLabV3+")
        model = smp.DeepLabV3Plus(
            encoder_name=backbone,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=input_channels,
            classes=num_classes,
        )
        return model
    except ImportError:
        logger.warning(
            "segmentation-models-pytorch not available — falling back to torchvision DeepLabV3"
        )

    # Torchvision fallback
    from torchvision.models.segmentation import deeplabv3_resnet50
    from torchvision.models.segmentation.deeplabv3 import DeepLabHead

    model = deeplabv3_resnet50(
        weights="COCO_WITH_VOC_LABELS_V1" if pretrained else None,
    )
    
    # Replace the classifier head for our custom number of classes (1)
    model.classifier = DeepLabHead(2048, num_classes)
    # Replace first conv layer for 6 input channels
    old_conv = model.backbone.conv1
    new_conv = nn.Conv2d(
        in_channels=input_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )

    # Initialize: copy RGB weights, random init extras
    with torch.no_grad():
        if old_conv.weight.shape[1] >= 3:
            new_conv.weight[:, :3] = old_conv.weight[:, :3]
        for c in range(3, input_channels):
            nn.init.normal_(new_conv.weight[:, c:c + 1], std=old_conv.weight.std().item())

    # DeepLabV3 uses aux classifier — we want per-pixel sigmoid output
    model.backbone.conv1 = new_conv
    return model


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

class Trainer:
    """
    Encapsulates training state, loop, validation, checkpointing, and logging.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        lr: float = DEFAULT_LR,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        epochs: int = DEFAULT_EPOCHS,
        device: str = "cuda",
        output_dir: str = "outputs/baseline",
        pos_weight: float = POS_WEIGHT_ROAD,
        use_amp: bool = True,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_amp = use_amp and device == "cuda"

        # Loss: BCE with class weighting
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=device),
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # Scheduler: Cosine annealing
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=epochs,
            eta_min=lr * 0.01,
        )

        # AMP scaler
        self.scaler = GradScaler(enabled=self.use_amp)

        # State
        self.current_epoch: int = 0
        self.best_val_iou: float = 0.0
        self.best_epoch: int = 0
        self.csv_path = self.output_dir / "metrics.csv"

        # Initialize CSV
        self._init_csv()

    def _init_csv(self) -> None:
        """Write CSV header."""
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss",
                "train_iou",
                "train_dice",
                "train_pixel_acc",
                "val_loss",
                "val_iou",
                "val_dice",
                "val_pixel_acc",
                "val_precision",
                "val_recall",
                "lr",
                "time_sec",
            ])

    def train(self) -> Dict[str, float]:
        """
        Run full training loop.

        Returns final validation metrics.
        """
        logger.info(
            "Training on %s for %d epochs — %d train batches, %d val batches",
            self.device, self.epochs,
            len(self.train_loader), len(self.val_loader),
        )
        logger.info("Baseline IoU target: >%.2f", TARGET_IOU)

        for epoch in range(1, self.epochs + 1):
            self.current_epoch = epoch
            t_start = time.time()

            # Train
            train_loss, train_metrics = self._train_one_epoch()

            # Validate
            val_loss, val_metrics = self._validate()

            # Scheduler step
            self.scheduler.step()

            # Checkpoint
            is_best = val_metrics["iou"] > self.best_val_iou
            if is_best:
                self.best_val_iou = val_metrics["iou"]
                self.best_epoch = epoch
                self._save_checkpoint("best_model.pth")

            elapsed = time.time() - t_start

            # Log to CSV
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch,
                    train_loss,
                    train_metrics.get("iou", 0),
                    train_metrics.get("dice", 0),
                    train_metrics.get("pixel_acc", 0),
                    val_loss,
                    val_metrics["iou"],
                    val_metrics["dice"],
                    val_metrics["pixel_acc"],
                    val_metrics["precision"],
                    val_metrics["recall"],
                    self.optimizer.param_groups[0]["lr"],
                    elapsed,
                ])

            # Console log
            logger.info(
                "Epoch %3d/%d | "
                "train_loss=%.4f train_iou=%.3f | "
                "val_loss=%.4f val_iou=%.3f val_dice=%.3f val_acc=%.3f | "
                "lr=%.2e | %.1fs %s",
                epoch, self.epochs,
                train_loss, train_metrics.get("iou", 0),
                val_loss, val_metrics["iou"], val_metrics["dice"], val_metrics["pixel_acc"],
                self.optimizer.param_groups[0]["lr"],
                elapsed,
                "★ BEST" if is_best else "",
            )

        # Save final checkpoint
        self._save_checkpoint("last_model.pth")

        # Final report
        logger.info("=" * 60)
        logger.info("Training complete — %d epochs", self.epochs)
        logger.info("Best IoU: %.4f (epoch %d)", self.best_val_iou, self.best_epoch)
        if self.best_val_iou >= TARGET_IOU:
            logger.info("✅ Baseline target met (IoU > %.2f)", TARGET_IOU)
        else:
            logger.info("⚠️  Below baseline target (IoU < %.2f) — review loss / data", TARGET_IOU)  # noqa
        logger.info("Metrics saved to: %s", self.csv_path)
        logger.info("=" * 60)

        return {"best_iou": self.best_val_iou, "best_epoch": self.best_epoch}

    def _train_one_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Single training epoch. Returns (avg_loss, metrics_dict)."""
        self.model.train()
        total_loss = 0.0
        all_metrics = []

        for batch_idx, (fused, mask) in enumerate(self.train_loader):
            fused = fused.to(self.device, non_blocking=True)
            mask = mask.to(self.device, non_blocking=True).float().unsqueeze(1)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.use_amp):
                output = self.model(fused)
                if isinstance(output, dict):
                    output = output["out"]
                loss = self.criterion(output, mask)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

            with torch.no_grad():
                metrics = compute_metrics(output.detach(), mask.squeeze(1).long())
                all_metrics.append(metrics)

        avg_loss = total_loss / max(len(self.train_loader), 1)
        avg_metrics = metrics_avg(all_metrics)
        return avg_loss, avg_metrics

    @torch.no_grad()
    def _validate(self) -> Tuple[float, Dict[str, float]]:
        """Validation pass. Returns (avg_loss, metrics_dict)."""
        self.model.eval()
        total_loss = 0.0
        all_metrics = []

        for fused, mask in self.val_loader:
            fused = fused.to(self.device, non_blocking=True)
            mask = mask.to(self.device, non_blocking=True).float().unsqueeze(1)

            output = self.model(fused)
            if isinstance(output, dict):
                output = output["out"]
            loss = self.criterion(output, mask)
            total_loss += loss.item()

            metrics = compute_metrics(output, mask.squeeze(1).long())
            all_metrics.append(metrics)

        avg_loss = total_loss / max(len(self.val_loader), 1)
        avg_metrics = metrics_avg(all_metrics)
        return avg_loss, avg_metrics

    def _save_checkpoint(self, filename: str) -> None:
        """Save model, optimizer, and training state."""
        path = self.output_dir / filename
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_iou": self.best_val_iou,
            "best_epoch": self.best_epoch,
        }
        torch.save(checkpoint, path)
        logger.debug("Checkpoint saved: %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 7 — DeepLabV3+ Baseline Training on Synthetic Tiles"
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY, help="Weight decay")
    parser.add_argument("--num-tiles", type=int, default=NUM_TILES, help="Number of synthetic tiles")
    parser.add_argument("--tile-size", type=int, default=TILE_SIZE, help="Tile size")
    parser.add_argument("--cache-dir", type=str, default="data/synthetic_tiles", help="Tile cache directory")
    parser.add_argument("--output-dir", type=str, default="outputs/baseline", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision")
    parser.add_argument("--pos-weight", type=float, default=POS_WEIGHT_ROAD, help="Positive class weight")
    args = parser.parse_args()

    # Device check
    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available — falling back to CPU")
        device_str = "cpu"

    logger.info("=" * 60)
    logger.info("Phase 7 — DeepLabV3+ Baseline Training")
    logger.info("  Batch size: %d", args.batch_size)
    logger.info("  Epochs: %d", args.epochs)
    logger.info("  LR: %.1e, Weight Decay: %.1e", args.lr, args.weight_decay)
    logger.info("  Pos weight (road): %.1f", args.pos_weight)
    logger.info("  Num tiles: %d, Tile size: %d", args.num_tiles, args.tile_size)
    logger.info("  Device: %s, AMP: %s", device_str, not args.no_amp)
    logger.info("  Output: %s", args.output_dir)
    logger.info("=" * 60)

    # Build dataloaders
    from dataset import RoadDataset

    train_ds = RoadDataset(
        tile_size=args.tile_size,
        num_tiles=args.num_tiles,
        split="train",
        augment=True,
        cache_dir=args.cache_dir,
    )
    val_ds = RoadDataset(
        tile_size=args.tile_size,
        num_tiles=args.num_tiles,
        split="val",
        augment=False,
        cache_dir=args.cache_dir,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device_str == "cuda"),
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device_str == "cuda"),
    )

    # Build model
    model = build_model(input_channels=INPUT_CHANNELS, num_classes=NUM_CLASSES)
    logger.info("Model: DeepLabV3+ with ResNet-50 backbone, %d input channels", INPUT_CHANNELS)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info("Trainable parameters: %d", param_count)

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        device=device_str,
        output_dir=args.output_dir,
        pos_weight=args.pos_weight,
        use_amp=not args.no_amp,
    )
    trainer.train()


if __name__ == "__main__":
    main()
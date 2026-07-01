#!/usr/bin/env python3
"""
ITEM 3: Corrected multi-epoch training pass.
=========================================================
- Uses OSMnx-based GT mask (3.49% density, fixed in dataset.py)
- Uses recalibrated pos_weight=27.0 (was 6.0)
- Runs 20 epochs (or as many as compute budget allows)
- Reports IoU, Precision, Recall, F1 on held-out val split
- Reports predicted road pixel % vs GT road pixel % on 3 sample tiles
- Saves PNG comparison: predicted mask vs GT side-by-side for 2 tiles

Usage:
    python calibration_train_corrected.py
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# --- Config ---
EPOCHS        = 20
BATCH_SIZE    = 2       # conservative for systems without large VRAM
LR            = 1e-4
POS_WEIGHT    = 27.0    # (1 - 0.035) / 0.035 = 27.57 -> 27.0
N_TRAIN_TILES = 100
N_VAL_TILES   = 25
TILE_SIZE     = 512
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR       = Path("part_a_vision/outputs/calibrated_training")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_PATH     = Path("part_a_vision/models/best_checkpoint.pth")
CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("ITEM 3: CORRECTED TRAINING RUN")
print(f"  Device: {DEVICE}")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  pos_weight: {POS_WEIGHT} (was 6.0)")
print(f"  Train tiles: {N_TRAIN_TILES} | Val tiles: {N_VAL_TILES}")
print(f"  GT mask density: ~3.49% (fixed, was ~29.5%)")
print("=" * 65)

# --- Dataset ---
from part_a_vision.dataset import RoadDataset
RoadDataset._osmnx_gt_mask_cache = None  # force reload

cache_dir_train = "part_a_vision/data/koramangala/train"
cache_dir_val   = "part_a_vision/data/koramangala/val"

train_ds = RoadDataset(
    tile_size=TILE_SIZE, num_tiles=N_TRAIN_TILES,
    split="train", augment=False,  # disable augmentation to avoid accidental occlusion altering mask stats
    cache_dir=cache_dir_train,
)
val_ds = RoadDataset(
    tile_size=TILE_SIZE, num_tiles=N_TRAIN_TILES,
    split="val", augment=False,
    cache_dir=cache_dir_val,
)

print(f"\n[Dataset] train={len(train_ds)} tiles, val={len(val_ds)} tiles")

# Verify density on first 5 train tiles
print("\n[DENSITY VERIFICATION - first 5 train tiles]")
densities = []
for i in range(min(5, len(train_ds))):
    real_idx = train_ds.indices[i]
    tile = train_ds.tiles[real_idx]
    seed = tile["seed"]
    cache_file = os.path.join(cache_dir_train, f"tile_{seed:04d}.npz")
    _, mask_raw = train_ds._load_or_generate(seed, cache_file)
    mask_arr = np.array(mask_raw)
    pct = float(mask_arr.sum()) / float(mask_arr.size) * 100
    densities.append(pct)
    print(f"  tile {i} (seed {seed}): {mask_arr.sum():,} road px / {mask_arr.size:,} total = {pct:.2f}%")
print(f"  Mean density: {np.mean(densities):.2f}%")

# --- DataLoaders ---
train_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=0, pin_memory=(DEVICE == "cuda"),
)
val_loader = torch.utils.data.DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=0, pin_memory=(DEVICE == "cuda"),
)

# --- Model ---
from part_a_vision.model import build_model
model = build_model(backbone="segformer_b3", input_channels=12, num_classes=1)
model = model.to(DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n[Model] segformer_b3 | {n_params:,} trainable parameters")

# --- Loss ---
from part_a_vision.loss import CombinedLoss
criterion = CombinedLoss(
    dice_weight=0.4, bce_weight=0.3, boundary_weight=0.2, conn_weight=0.1,
    pos_weight=POS_WEIGHT, use_focal=True
)

# --- Optimizer ---
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.01)

# --- AMP ---
use_amp = (DEVICE == "cuda")
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

# --- Metrics ---
def compute_metrics(logits, targets, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).long()
    targets = targets.long()
    tp = (preds * targets).sum().float()
    fp = (preds * (1 - targets)).sum().float()
    fn = ((1 - preds) * targets).sum().float()
    union = (preds + targets).clamp(0, 1).sum().float()
    iou = (tp / (union + 1e-8)).item()
    precision = (tp / (tp + fp + 1e-8)).item()
    recall = (tp / (tp + fn + 1e-8)).item()
    f1 = (2 * precision * recall / (precision + recall + 1e-8))
    return {"iou": iou, "precision": precision, "recall": recall, "f1": f1}

# --- Training loop ---
best_iou = 0.0
best_epoch = 0
train_log = []

print(f"\n[Training] Starting {EPOCHS} epochs...")
print(f"{'Epoch':>5} {'Train_Loss':>12} {'Train_IoU':>10} {'Val_Loss':>12} {'Val_IoU':>10} {'Val_F1':>8} {'Time':>7}")
print("-" * 70)

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()

    # --- Train ---
    model.train()
    t_loss_sum, t_iou_sum, t_batches = 0.0, 0.0, 0
    for fused, mask in train_loader:
        fused = fused.to(DEVICE, non_blocking=True)
        mask  = mask.to(DEVICE, non_blocking=True).float().unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            out = model(fused)
            if isinstance(out, dict): out = out["out"]
            loss, _ = criterion(out, mask)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        t_loss_sum += loss.item()
        m = compute_metrics(out.detach(), mask.squeeze(1))
        t_iou_sum += m["iou"]
        t_batches += 1

    scheduler.step()
    t_loss = t_loss_sum / max(t_batches, 1)
    t_iou  = t_iou_sum  / max(t_batches, 1)

    # --- Validate ---
    model.eval()
    v_loss_sum, v_metrics_list, v_batches = 0.0, [], 0
    with torch.no_grad():
        for fused, mask in val_loader:
            fused = fused.to(DEVICE, non_blocking=True)
            mask  = mask.to(DEVICE, non_blocking=True).float().unsqueeze(1)
            out = model(fused)
            if isinstance(out, dict): out = out["out"]
            loss, _ = criterion(out, mask)
            v_loss_sum += loss.item()
            v_metrics_list.append(compute_metrics(out, mask.squeeze(1)))
            v_batches += 1

    v_loss = v_loss_sum / max(v_batches, 1)
    v_iou  = np.mean([m["iou"]  for m in v_metrics_list])
    v_prec = np.mean([m["precision"] for m in v_metrics_list])
    v_rec  = np.mean([m["recall"]    for m in v_metrics_list])
    v_f1   = np.mean([m["f1"]        for m in v_metrics_list])
    elapsed = time.time() - t0

    is_best = v_iou > best_iou
    if is_best:
        best_iou = v_iou
        best_epoch = epoch
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_iou": best_iou,
        }, CKPT_PATH)
        flag = " ** BEST **"
    else:
        flag = ""

    print(f"{epoch:>5} {t_loss:>12.4f} {t_iou:>10.4f} {v_loss:>12.4f} {v_iou:>10.4f} {v_f1:>8.4f} {elapsed:>6.1f}s{flag}")
    train_log.append({
        "epoch": epoch, "train_loss": t_loss, "train_iou": t_iou,
        "val_loss": v_loss, "val_iou": v_iou, "val_prec": v_prec,
        "val_rec": v_rec, "val_f1": v_f1
    })

print("-" * 70)
print(f"\n[RESULT] Best val IoU = {best_iou:.4f} at epoch {best_epoch}")
print(f"         Checkpoint saved to: {CKPT_PATH}")

# --- Per-tile density gap analysis on 3 val tiles ---
print("\n" + "=" * 65)
print("ITEM 3: PREDICTED VS GT ROAD PIXEL % on 3 val tiles")
print("=" * 65)

model.eval()
val_ds_check = RoadDataset(
    tile_size=TILE_SIZE, num_tiles=N_TRAIN_TILES,
    split="val", augment=False,
    cache_dir=cache_dir_val,
)

fig, axes = plt.subplots(3, 4, figsize=(20, 15))
fig.suptitle("ITEM 3: Predicted vs GT Road Masks (3 Val Tiles)\nCorrected training: pos_weight=27.0, density=3.49%", fontsize=13)

with torch.no_grad():
    for tile_idx in range(3):
        fused_t, mask_t = val_ds_check[tile_idx]
        fused_in = fused_t.unsqueeze(0).to(DEVICE)
        out = model(fused_in)
        if isinstance(out, dict): out = out["out"]
        prob = torch.sigmoid(out).squeeze().cpu().numpy()
        pred_mask = (prob > 0.5).astype(np.uint8)
        gt_mask = mask_t.numpy().astype(np.uint8)

        pred_pct = float(pred_mask.sum()) / float(pred_mask.size) * 100
        gt_pct   = float(gt_mask.sum())   / float(gt_mask.size)   * 100
        gap      = abs(pred_pct - gt_pct)

        # Compute per-tile IoU
        inter = (pred_mask & gt_mask).sum()
        union = (pred_mask | gt_mask).sum()
        t_iou = inter / (union + 1e-8)

        print(f"  Tile {tile_idx}: GT={gt_pct:.2f}%  Pred={pred_pct:.2f}%  Gap={gap:.2f}%  IoU={t_iou:.4f}")

        # Visualization
        r = tile_idx
        # Col 0: optical false color (NIR-R-G using channels 2,1,0)
        fc = np.stack([fused_t[2], fused_t[1], fused_t[0]], axis=-1)
        fc = np.clip(fc, 0, 1)
        axes[r][0].imshow(fc)
        axes[r][0].set_title(f"Tile {tile_idx} - Optical (NIR-R-G)")
        axes[r][0].axis("off")

        axes[r][1].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
        axes[r][1].set_title(f"GT mask ({gt_pct:.2f}% road)")
        axes[r][1].axis("off")

        axes[r][2].imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
        axes[r][2].set_title(f"Predicted ({pred_pct:.2f}% road, IoU={t_iou:.3f})")
        axes[r][2].axis("off")

        # Difference map
        diff = np.zeros((*gt_mask.shape, 3), dtype=np.float32)
        diff[..., 0] = (pred_mask & ~gt_mask.astype(bool)).astype(float)  # FP = red
        diff[..., 1] = (gt_mask & ~pred_mask.astype(bool)).astype(float)  # FN = green
        diff[..., 2] = (pred_mask & gt_mask.astype(bool)).astype(float)   # TP = blue
        axes[r][3].imshow(diff)
        axes[r][3].set_title("FP=red | FN=green | TP=blue")
        axes[r][3].axis("off")

plt.tight_layout()
vis_path = OUT_DIR / "item3_pred_vs_gt_comparison.png"
plt.savefig(str(vis_path), dpi=100, bbox_inches="tight")
plt.close()
print(f"\n[SAVED] Visualization: {vis_path}")

# --- Final summary ---
print("\n" + "=" * 65)
print("ITEM 3: FINAL TRAINING SUMMARY (real numbers)")
print("=" * 65)
final = train_log[-1]
print(f"  Epochs actually completed: {EPOCHS}")
print(f"  Best val IoU:   {best_iou:.4f}  (epoch {best_epoch})")
print(f"  Final val IoU:  {final['val_iou']:.4f}")
print(f"  Final val Prec: {final['val_prec']:.4f}")
print(f"  Final val Rec:  {final['val_rec']:.4f}")
print(f"  Final val F1:   {final['val_f1']:.4f}")

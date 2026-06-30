#!/usr/bin/env python3
"""
Phase 7 — Task 5: Baseline Evaluation & Inference
==================================================
Loads the best trained DeepLabV3+ model, runs inference on the validation
set, and computes detailed pixel-level and topology metrics:

  Pixel-level:
    - IoU (Jaccard)
    - Dice (F1)
    - Pixel Accuracy
    - Precision / Recall

  Topology (skeleton-based):
    - Skeleton IoU: IoU of morphological skeletons
    - Connected Component count: #CC(pred) vs #CC(gt) for connectivity analysis
    - Skeleton Precision / Recall

  Per-tile breakdown saved to CSV; visual overlays saved as PNG.

Place at: part_a_vision/evaluate_baseline.py

Usage:
  python evaluate_baseline.py --checkpoint outputs/baseline/best_model.pth --output-dir outputs/eval
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from skimage.morphology import skeletonize

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("evaluate_baseline")
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
INPUT_CHANNELS: int = 6
NUM_CLASSES: int = 1
THRESHOLD: float = 0.5
MIN_ROAD_PIXELS: int = 50  # ignore tiles with too few road pixels


# ---------------------------------------------------------------------------
# Skeleton & Topology Metrics
# ---------------------------------------------------------------------------

def compute_skeleton(mask: np.ndarray) -> np.ndarray:
    """
    Compute morphological skeleton of a binary mask.

    Uses skimage skeletonize (Zhang-Suen thinning) which produces
    a one-pixel-wide centerline preserving topology.
    """
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=np.uint8)
    return skeletonize(mask.astype(bool)).astype(np.uint8)


def skeleton_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    IoU of morphological skeletons — measures topological overlap.
    Both inputs should be binary {0, 1} masks.
    """
    pred_skel = compute_skeleton(pred)
    gt_skel = compute_skeleton(gt)

    intersection = (pred_skel & gt_skel).sum()
    union = (pred_skel | gt_skel).sum()

    if union == 0:
        return 1.0  # both empty — perfect match
    return float(intersection) / float(union)


def skeleton_precision_recall(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    """
    Skeleton precision & recall.

    Precision: fraction of predicted skeleton within 2px of ground-truth skeleton.
    Recall: fraction of ground-truth skeleton within 2px of predicted skeleton.
    """
    pred_skel = compute_skeleton(pred)
    gt_skel = compute_skeleton(gt)

    # Dilate for tolerance band (2px buffer)
    struct = np.ones((5, 5), dtype=np.uint8)  # 2px radius
    gt_dilated = ndimage.binary_dilation(gt_skel, structure=struct, iterations=1)

    if pred_skel.sum() == 0:
        precision = 0.0 if gt_skel.sum() > 0 else 1.0
    else:
        precision = (pred_skel & gt_dilated).sum() / pred_skel.sum()

    pred_dilated = ndimage.binary_dilation(pred_skel, structure=struct, iterations=1)

    if gt_skel.sum() == 0:
        recall = 1.0 if pred_skel.sum() == 0 else 0.0
    else:
        recall = (gt_skel & pred_dilated).sum() / gt_skel.sum()

    return float(precision), float(recall)


def connected_component_count(mask: np.ndarray) -> int:
    """
    Count connected components (8-connectivity) in binary mask.
    """
    if mask.sum() == 0:
        return 0
    _, num_cc = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    return num_cc


# ---------------------------------------------------------------------------
# Pixel-Level Metrics
# ---------------------------------------------------------------------------

def compute_pixel_metrics(
    pred: np.ndarray, gt: np.ndarray
) -> Dict[str, float]:
    """
    Compute IoU, Dice, pixel accuracy, precision, recall.

    Parameters
    ----------
    pred : (H, W) uint8 — binary prediction {0, 1}
    gt : (H, W) uint8 — binary ground truth {0, 1}

    Returns
    -------
    dict with: iou, dice, pixel_acc, precision, recall
    """
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    iou = float(intersection) / max(union, 1)

    dice = (2.0 * intersection) / max(pred.sum() + gt.sum(), 1)

    correct = (pred == gt).sum()
    total = pred.size
    pixel_acc = float(correct) / total

    tp = float(intersection)
    fp = float((pred & (1 - gt)).sum())
    fn = float(((1 - pred) & gt).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    return {
        "iou": iou,
        "dice": dice,
        "pixel_acc": pixel_acc,
        "precision": precision,
        "recall": recall,
    }


def compute_width_split_iou(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """Split road pixels into thin (<3px) and major (>5px) using the skeleton."""
    from skimage.morphology import medial_axis
    from scipy.ndimage import distance_transform_edt
    
    skel, distance = medial_axis(gt, return_distance=True)
    if skel.sum() == 0:
        return {'thin_iou': float('nan'), 'major_iou': float('nan')}
        
    width_at_skel = (2 * distance) * skel
    dist_to_skel, indices = distance_transform_edt(1 - skel, return_indices=True)
    width_map = width_at_skel[indices[0], indices[1]]
    
    thin_mask = width_map <= 3.0
    major_mask = width_map > 5.0
    
    def _iou(mask):
        if mask.sum() == 0: return float('nan')
        p = pred[mask].astype(np.int32)
        g = gt[mask].astype(np.int32)
        tp = (p * g).sum()
        fp = (p * (1 - g)).sum()
        fn = ((1 - p) * g).sum()
        if (tp + fp + fn) == 0: return float('nan')
        return float(tp / (tp + fp + fn + 1e-8))
        
    return {
        'thin_iou': _iou(thin_mask),
        'major_iou': _iou(major_mask)
    }

def compute_edge_f1(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.3) -> float:
    """Compute F1 score on road boundaries."""
    from scipy.ndimage import sobel
    def _sobel_edges(mask_arr):
        m = mask_arr.astype(np.float32)
        gx = sobel(m, axis=0)
        gy = sobel(m, axis=1)
        mag = np.hypot(gx, gy)
        max_mag = mag.max()
        if max_mag < 1e-8: return np.zeros_like(mask_arr, dtype=np.uint8)
        return (mag > threshold * max_mag).astype(np.uint8)
        
    pred_edges = _sobel_edges(pred)
    gt_edges = _sobel_edges(gt)
    if gt_edges.sum() == 0:
        return float('nan')
        
    tp = (pred_edges & gt_edges).sum()
    fp = (pred_edges & (1 - gt_edges)).sum()
    fn = ((1 - pred_edges) & gt_edges).sum()
    return float((2 * tp) / (2 * tp + fp + fn + 1e-8))

# ---------------------------------------------------------------------------
# Model Loader
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: str = "cpu", backbone: str = "segformer_b3") -> torch.nn.Module:
    """
    Load trained model from checkpoint using the model factory.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    from model import build_model
    model = build_model(backbone=backbone, input_channels=INPUT_CHANNELS, num_classes=NUM_CLASSES)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Loaded %s from checkpoint (epoch %d)", backbone, checkpoint.get("epoch", -1))
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(
    model: torch.nn.Module,
    fused: torch.Tensor,
    threshold: float = THRESHOLD,
    device: str = "cpu",
    return_attention: bool = False,
):
    """
    Run inference on a single fused tile.
    """
    batch = fused.unsqueeze(0).to(device)
    attention_maps = None
    
    if return_attention and "return_attention" in model.forward.__code__.co_varnames:
        output, attention_maps = model(batch, return_attention=True)
    else:
        output = model(batch)
        
    if isinstance(output, dict):
        output = output["out"]
        
    probs = torch.sigmoid(output).squeeze(0).squeeze(0).cpu().numpy()
    mask = (probs > threshold).astype(np.uint8)
    
    if return_attention:
        return mask, attention_maps
    return mask


# ---------------------------------------------------------------------------
# Main Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cpu",
    output_dir: Optional[str] = None,
    save_visuals: bool = False,
    threshold: float = THRESHOLD,
) -> Dict[str, float]:
    """
    Full evaluation over a DataLoader.

    Returns aggregated metrics across all tiles.
    """
    all_metrics: List[Dict[str, float]] = []
    per_tile_rows: List[Dict] = []
    skipped = 0

    for idx, (fused, mask_gt) in enumerate(dataloader):
        # fused: (B, C, H, W), mask_gt: (B, H, W)
        for b in range(fused.shape[0]):
            tile = fused[b]
            gt = mask_gt[b].numpy().astype(np.uint8)

            # Skip tiles with too few road pixels (non-informative)
            if gt.sum() < MIN_ROAD_PIXELS:
                skipped += 1
                continue

            # Predict
            pred_out = predict(model, tile, threshold=threshold, device=device, return_attention=save_visuals)
            
            if isinstance(pred_out, tuple):
                pred, att_maps = pred_out
            else:
                pred, att_maps = pred_out, None

            # Pixel metrics
            pixel = compute_pixel_metrics(pred, gt)

            # Topology metrics
            skel_iou = skeleton_iou(pred, gt)
            skel_prec, skel_rec = skeleton_precision_recall(pred, gt)
            cc_pred = connected_component_count(pred)
            cc_gt = connected_component_count(gt)

            # Custom metrics
            width_metrics = compute_width_split_iou(pred, gt)
            edge_f1 = compute_edge_f1(pred, gt)

            metrics = {
                **pixel,
                "skeleton_iou": skel_iou,
                "skeleton_precision": skel_prec,
                "skeleton_recall": skel_rec,
                "cc_pred": cc_pred,
                "cc_gt": cc_gt,
                "thin_iou": width_metrics["thin_iou"],
                "major_iou": width_metrics["major_iou"],
                "edge_f1": edge_f1,
            }
            all_metrics.append(metrics)

            per_tile_rows.append({
                "tile": idx * dataloader.batch_size + b,
                **metrics,
            })

            # Visual overlay
            if save_visuals and output_dir:
                _save_overlay(pred, gt, output_dir, idx * dataloader.batch_size + b, attention_map=att_maps[-1] if att_maps else None)

    if skipped:
        logger.info("Skipped %d tiles with < %d road pixels", skipped, MIN_ROAD_PIXELS)

    # Aggregate
    agg = _aggregate(all_metrics)

    # Save per-tile CSV
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "per_tile_metrics.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=per_tile_rows[0].keys())
            writer.writeheader()
            writer.writerows(per_tile_rows)
        logger.info("Per-tile metrics saved: %s", csv_path)

    return agg


def _aggregate(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Compute mean and std across tiles."""
    if not metrics_list:
        return {}

    keys = metrics_list[0].keys()
    agg = {}
    for k in keys:
        values = [m[k] for m in metrics_list]
        agg[k] = np.mean(values)
        agg[f"{k}_std"] = np.std(values)

    # CC ratio (pred / gt) — measures fragmentation
    cc_pred_vals = [m["cc_pred"] for m in metrics_list]
    cc_gt_vals = [m["cc_gt"] for m in metrics_list]
    agg["cc_ratio"] = np.mean([p / max(g, 1) for p, g in zip(cc_pred_vals, cc_gt_vals)])
    agg["cc_ratio_std"] = np.std([p / max(g, 1) for p, g in zip(cc_pred_vals, cc_gt_vals)])

    return agg


def _save_overlay(
    pred: np.ndarray,
    gt: np.ndarray,
    output_dir: str,
    tile_idx: int,
    attention_map = None,
) -> None:
    """Save RGB overlay: green=TP, red=FP, blue=FN, black=TN."""
    try:
        from PIL import Image

        h, w = pred.shape
        overlay = np.zeros((h, w, 3), dtype=np.uint8)

        tp = (pred == 1) & (gt == 1)    # green
        fp = (pred == 1) & (gt == 0)    # red
        fn = (pred == 0) & (gt == 1)    # blue

        overlay[tp] = [0, 255, 0]
        overlay[fp] = [255, 0, 0]
        overlay[fn] = [0, 0, 255]

        vis_dir = os.path.join(output_dir, "visuals")
        os.makedirs(vis_dir, exist_ok=True)
        Image.fromarray(overlay).save(os.path.join(vis_dir, f"tile_{tile_idx:04d}.png"))
        
        # Save attention map if available
        if attention_map is not None:
            import matplotlib.pyplot as plt
            att = F.interpolate(attention_map, size=(h, w), mode="bilinear", align_corners=False)
            att = att.squeeze().cpu().numpy()
            plt.imsave(os.path.join(vis_dir, f"tile_{tile_idx:04d}_attention.png"), att, cmap='jet')
            
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Report Formatter
# ---------------------------------------------------------------------------

def format_report(metrics: Dict[str, float]) -> str:
    """Format evaluation metrics as a readable report string."""
    lines = []
    lines.append("=" * 60)
    lines.append("EVALUATION REPORT")
    lines.append("=" * 60)
    lines.append("")
    lines.append("— Pixel-Level Metrics —")
    lines.append(f"  IoU (Jaccard)         : {metrics.get('iou', 0):.4f} ± {metrics.get('iou_std', 0):.4f}")
    lines.append(f"  Dice (F1)              : {metrics.get('dice', 0):.4f} ± {metrics.get('dice_std', 0):.4f}")
    lines.append(f"  Pixel Accuracy         : {metrics.get('pixel_acc', 0):.4f} ± {metrics.get('pixel_acc_std', 0):.4f}")
    lines.append(f"  Precision (road)       : {metrics.get('precision', 0):.4f} ± {metrics.get('precision_std', 0):.4f}")
    lines.append(f"  Recall (road)          : {metrics.get('recall', 0):.4f} ± {metrics.get('recall_std', 0):.4f}")
    lines.append("")
    lines.append("— Width Breakdown —")
    lines.append(f"  Thin Road IoU (<3px)   : {metrics.get('thin_iou', 0):.4f} ± {metrics.get('thin_iou_std', 0):.4f}")
    lines.append(f"  Major Road IoU (>5px)  : {metrics.get('major_iou', 0):.4f} ± {metrics.get('major_iou_std', 0):.4f}")
    lines.append("")
    lines.append("— Topology Metrics —")
    lines.append(f"  Skeleton IoU           : {metrics.get('skeleton_iou', 0):.4f} ± {metrics.get('skeleton_iou_std', 0):.4f}")
    lines.append(f"  Skeleton Precision     : {metrics.get('skeleton_precision', 0):.4f} ± {metrics.get('skeleton_precision_std', 0):.4f}")
    lines.append(f"  Skeleton Recall        : {metrics.get('skeleton_recall', 0):.4f} ± {metrics.get('skeleton_recall_std', 0):.4f}")
    lines.append(f"  Edge F1 (Boundary)     : {metrics.get('edge_f1', 0):.4f} ± {metrics.get('edge_f1_std', 0):.4f}")
    lines.append(f"  CC Count (pred)        : {metrics.get('cc_pred', 0):.1f} ± {metrics.get('cc_pred_std', 0):.1f}")
    lines.append(f"  CC Count (gt)          : {metrics.get('cc_gt', 0):.1f} ± {metrics.get('cc_gt_std', 0):.1f}")
    lines.append(f"  CC Ratio (pred/gt)     : {metrics.get('cc_ratio', 0):.4f} ± {metrics.get('cc_ratio_std', 0):.4f}")
    lines.append("")
    baseline_met = metrics.get('iou', 0) >= 0.30
    lines.append(f"Baseline target (IoU > 0.30): {'✅ MET' if baseline_met else '⚠️  NOT MET'}")
    lines.append("=" * 60)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 7/11 — Evaluate trained baseline/SegFormer"
    )
    parser.add_argument("--backbone", type=str, default="segformer_b3", help="Model backbone")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--output-dir", type=str, default="outputs/eval", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--num-tiles", type=int, default=200, help="Number of synthetic tiles")
    parser.add_argument("--tile-size", type=int, default=512, help="Tile size")
    parser.add_argument("--cache-dir", type=str, default="data/synthetic_tiles", help="Tile cache")
    parser.add_argument("--save-visuals", action="store_true", help="Save per-tile overlay PNGs")
    parser.add_argument("--threshold", type=float, default=THRESHOLD, help="Sigmoid threshold")
    parser.add_argument("--force-occlusion", action="store_true", help="Phase 12: Force canopy occlusion on all tiles to test CBAM")
    args = parser.parse_args()

    # Device
    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available — using CPU")
        device_str = "cpu"

    logger.info("Loading model from: %s", args.checkpoint)
    model = load_model(args.checkpoint, device=device_str, backbone=args.backbone)
    logger.info("Model loaded — running on %s", device_str)

    # Build validation dataloader
    from dataset import RoadDataset

    val_ds = RoadDataset(
        tile_size=args.tile_size,
        num_tiles=args.num_tiles,
        split="val",
        augment=False,
        cache_dir=args.cache_dir,
        force_occlusion=args.force_occlusion,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    logger.info("Validation set: %d tiles", len(val_ds))

    # Run evaluation
    metrics = evaluate(
        model=model,
        dataloader=val_loader,
        device=device_str,
        output_dir=args.output_dir,
        save_visuals=args.save_visuals,
        threshold=args.threshold,
    )

    # Print report
    report = format_report(metrics)
    print("\n" + report + "\n")

    # Save report
    report_path = os.path.join(args.output_dir, "evaluation_report.txt")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report)
    logger.info("Report saved: %s", report_path)

    # Summary JSON
    import json
    json_path = os.path.join(args.output_dir, "metrics_summary.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("JSON summary saved: %s", json_path)


if __name__ == "__main__":
    main()
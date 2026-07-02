"""
ITEM 1 — Measure existing road density before fix.
Shows: current GT mask density, and per-tile density in train split.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from pathlib import Path

print("=" * 65)
print("ITEM 1: MEASURING CURRENT ROAD DENSITY (BEFORE FIX)")
print("=" * 65)

# --- 1. Measure the raw OSMnx GT mask ---
gt_mask_path = Path("part_a_vision/data/koramangala/osmnx_gt_mask.npy")
if gt_mask_path.exists():
    gt = np.load(str(gt_mask_path))
    H, W = gt.shape
    road_px = int(gt.sum())
    total_px = H * W
    pct = road_px / total_px * 100
    print(f"\n[GT MASK] File: {gt_mask_path}")
    print(f"  Shape: {H}x{W} = {total_px:,} total pixels")
    print(f"  Road pixels: {road_px:,}")
    print(f"  Road density: {pct:.2f}%")
    print(f"  Road width used to generate (ROAD_WIDTH_PX): see synthetic_tile.py line 125")
else:
    print(f"  [WARNING] GT mask not found at {gt_mask_path}")

# --- 2. Measure 10 existing train tiles ---
train_dir = Path("part_a_vision/data/koramangala/train")
tiles = sorted(train_dir.glob("tile_*.npz"))[:10]

print(f"\n[TRAIN TILES] Measuring first {len(tiles)} tiles:")
print(f"  {'Tile':<15} {'Road_px':>10} {'Total_px':>10} {'Density%':>10}")
print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10}")

densities = []
for tile_path in tiles:
    data = np.load(str(tile_path))
    mask = data['gt_mask']   # (H, W) uint8
    road_px = int(mask.sum())
    total_px = mask.size
    pct = road_px / total_px * 100
    densities.append(pct)
    print(f"  {tile_path.name:<15} {road_px:>10,} {total_px:>10,} {pct:>9.2f}%")

if densities:
    print(f"\n  MEAN road density across these 10 tiles: {np.mean(densities):.2f}%")
    print(f"  (Real Koramangala ground truth target: ~3.5%)")
    print(f"  (Acceptable synthetic range: 3–8%)")

print("\n[DIAGNOSIS]")
print(f"  ROAD_WIDTH_PX = 2  (in synthetic_tile.py line 125)")
print(f"  This may still produce high density IF the OSMnx graph has many edges")
print(f"  and they densely cover the tile at 512x512 resolution.")
print(f"  The FIX: keep road_width_px=2 but dilate the rasterized mask by 0 extra")
print(f"  (it's already thin), and instead DELETE the cached osmnx_gt_mask.npy so")
print(f"  it regenerates fresh with corrected parameters.")
print(f"  But if density is already fine, the issue is something else.")
print("=" * 65)

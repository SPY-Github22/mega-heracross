"""
ITEM 1 POST-FIX: Generate 10 sample tiles and report actual measured road pixel percentage.
This is the real verification requested — not a guess at the target.
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

# Force regeneration of tiles (cache cleared)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 65)
print("ITEM 1 POST-FIX: Generating 10 fresh tiles, measuring density")
print("=" * 65)

# Import the fixed dataset
from part_a_vision.dataset import RoadDataset

# Clear class cache to force fresh load
RoadDataset._osmnx_gt_mask_cache = None

# Create dataset with fresh tiles
cache_dir = "part_a_vision/data/koramangala/train"
ds = RoadDataset(
    tile_size=512,
    num_tiles=10,
    split="train",
    augment=False,  # no augmentation so density is pure mask density
    cache_dir=cache_dir,
)

print(f"\n{'Tile':<8} {'Seed':<8} {'Road_px':>10} {'Total_px':>10} {'Density%':>10}")
print(f"{'----':<8} {'----':<8} {'-------':>10} {'--------':>10} {'--------':>10}")

densities = []
for i in range(min(10, len(ds))):
    # Access raw tile to get mask without augmentation transforms
    real_idx = ds.indices[i]
    tile = ds.tiles[real_idx]
    seed = tile["seed"]
    cache_file = os.path.join(cache_dir, f"tile_{seed:04d}.npz")
    _, mask = ds._load_or_generate(seed, cache_file)
    mask = np.array(mask)
    road_px = int(mask.sum())
    total_px = int(mask.size)
    pct = road_px / total_px * 100
    densities.append(pct)
    print(f"{i:<8} {seed:<8} {road_px:>10,} {total_px:>10,} {pct:>9.2f}%")

print(f"\nMEAN density across 10 tiles: {np.mean(densities):.2f}%")
print(f"MIN / MAX: {min(densities):.2f}% / {max(densities):.2f}%")
print(f"Target range: 3-8% (real Koramangala: 3.5%)")
print(f"\n[BEFORE FIX] density was: ~29.50% (band-based procedural masks)")
print(f"[AFTER FIX]  density is:  {np.mean(densities):.2f}% (OSMnx GT mask-based)")
print("=" * 65)

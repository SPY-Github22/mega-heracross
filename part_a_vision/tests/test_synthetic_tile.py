# part_a_vision/tests/test_synthetic_tile.py
# Exit criterion tests for Phase 2.
#
# Run from repo root:
#   python part_a_vision/tests/test_synthetic_tile.py
#
# What this verifies:
#   1. Generator produces correct shapes and dtypes
#   2. Spectral signatures are physically plausible (roads < veg in NIR)
#   3. Cloud occlusion: SAR is UNCHANGED, optical brightness INCREASES
#   4. Canopy occlusion: road pixels' optical values shift toward vegetation signature
#   5. road_mask.npy and meta.json pass contract validation
#   6. Visualization PNG is saved and non-empty
#   7. Dataset generation produces correctly structured files
#   8. Determinism: same seed → identical tile

import os
import sys
import json
import numpy as np
from pathlib import Path

_tests_dir  = Path(__file__).resolve().parent
_part_a_dir = _tests_dir.parent
_repo_root  = _part_a_dir.parent
sys.path.insert(0, str(_repo_root))

from part_a_vision.synthetic_tile import (
    SyntheticTileGenerator, visualize_tile, generate_dataset, save_demo_output
)
from part_a_vision.output_writer import load_and_verify
from shared.config import ROAD_MASK_PATH, META_PATH

# ── Test runner ───────────────────────────────────────────────────────────────
_pass  = 0
_fail  = 0
_gen   = None  # shared generator instance (expensive to create each time)

def check(condition: bool, label: str, detail: str = "") -> None:
    global _pass, _fail
    if condition:
        print(f"  ✓ PASS  [{label}]")
        _pass += 1
    else:
        print(f"  ✗ FAIL  [{label}]")
        if detail:
            print(f"         {detail}")
        _fail += 1

def get_generator() -> SyntheticTileGenerator:
    global _gen
    if _gen is None:
        print("  [setup] Creating SyntheticTileGenerator (downloads OSMnx if needed)...")
        _gen = SyntheticTileGenerator()
    return _gen


# ════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("  Phase 2 Exit Criterion Tests — synthetic_tile.py")
print("=" * 60)


# ── Section 1: Clean tile (no occlusion) ─────────────────────────────────────
print("\n[Section 1: Clean tile — shape, dtype, value range]")

gen  = get_generator()
tile = gen.generate(seed=42, occlusion_type='none', occlusion_fraction=0.0)

check(
    tile.optical.ndim == 3 and tile.optical.shape[0] == 4,
    "optical is (4, H, W)",
    f"Got shape: {tile.optical.shape}"
)
check(
    tile.optical.dtype == np.float32,
    "optical dtype is float32",
    f"Got: {tile.optical.dtype}"
)
check(
    tile.optical.shape[1] == tile.optical.shape[2] == 512,
    "optical is 512×512",
    f"Got: {tile.optical.shape[1]}×{tile.optical.shape[2]}"
)
check(
    float(tile.optical.min()) >= 0.0 and float(tile.optical.max()) <= 1.0,
    "optical values in [0, 1]",
    f"Got: [{tile.optical.min():.4f}, {tile.optical.max():.4f}]"
)
check(
    tile.sar.ndim == 3 and tile.sar.shape[0] == 2,
    "SAR is (2, H, W)",
    f"Got shape: {tile.sar.shape}"
)
check(
    tile.sar.dtype == np.float32,
    "SAR dtype is float32",
    f"Got: {tile.sar.dtype}"
)
check(
    float(tile.sar.min()) >= 0.0 and float(tile.sar.max()) <= 1.0,
    "SAR values in [0, 1]",
    f"Got: [{tile.sar.min():.4f}, {tile.sar.max():.4f}]"
)
check(
    tile.gt_mask.ndim == 2 and tile.gt_mask.dtype == np.uint8,
    "gt_mask is 2D uint8",
    f"Got: {tile.gt_mask.shape}, {tile.gt_mask.dtype}"
)
check(
    set(np.unique(tile.gt_mask).tolist()).issubset({0, 1}),
    "gt_mask values in {0, 1}",
    f"Got unique values: {np.unique(tile.gt_mask).tolist()}"
)
check(
    tile.gt_mask.sum() > 100,
    "gt_mask contains road pixels (sum > 100)",
    f"Got sum: {tile.gt_mask.sum()}"
)
check(
    tile.cloud_mask.sum() == 0,
    "cloud_mask is all-zero for 'none' occlusion",
    f"Got sum: {tile.cloud_mask.sum()}"
)


# ── Section 2: Spectral plausibility ─────────────────────────────────────────
print("\n[Section 2: Spectral signatures — physical plausibility]")

# Roads should be darker than background in all bands (asphalt absorbs)
road_px = tile.gt_mask.astype(bool)
bg_px   = ~road_px

for band_idx, band_name in enumerate(['Green', 'Red', 'NIR', 'SWIR']):
    road_mean = float(tile.optical[band_idx][road_px].mean())
    bg_mean   = float(tile.optical[band_idx][bg_px].mean())
    check(
        road_mean < bg_mean,
        f"{band_name}: road reflectance < background reflectance",
        f"Road mean={road_mean:.3f}, Background mean={bg_mean:.3f}"
    )

# NDVI: roads should have low NDVI, vegetation (if present) should have high NDVI
nir  = tile.optical[2]
red  = tile.optical[1]
ndvi = (nir - red) / (nir + red + 1e-8)
road_ndvi = float(ndvi[road_px].mean())
check(
    road_ndvi < 0.15,
    f"Road NDVI is low (< 0.15, got {road_ndvi:.3f})",
    "Roads should have NDVI near zero (asphalt, no vegetation)"
)

# SAR roads should be darker than background (specular reflection)
road_vv_mean = float(tile.sar[0][road_px].mean())
bg_vv_mean   = float(tile.sar[0][bg_px].mean())
check(
    road_vv_mean < bg_vv_mean,
    f"SAR VV: road backscatter < background ({road_vv_mean:.3f} < {bg_vv_mean:.3f})",
    "Roads should be dark in SAR (specular reflection away from antenna)"
)


# ── Section 3: Cloud occlusion physics ───────────────────────────────────────
print("\n[Section 3: Cloud occlusion — SAR unchanged, optical brightened]")

tile_clean = gen.generate(seed=99, occlusion_type='none', occlusion_fraction=0.0)
tile_cloud = gen.generate(seed=99, occlusion_type='cloud', occlusion_fraction=0.50)

cloud_px = tile_cloud.cloud_mask.astype(bool)
n_cloud_px = cloud_px.sum()

check(
    n_cloud_px > 100,
    f"Cloud mask has > 100 pixels ({n_cloud_px:,} cloud pixels)",
)

# SAR MUST be identical under cloud mask (physics: radar penetrates clouds)
sar_diff = np.abs(tile_clean.sar - tile_cloud.sar)
max_sar_diff = float(sar_diff.max())
check(
    max_sar_diff < 1e-5,
    f"SAR is unchanged under cloud cover (max diff = {max_sar_diff:.2e})",
    "SAR should be physically identical before/after cloud application"
)

# Optical MUST be brighter in cloud regions
if n_cloud_px > 0:
    clean_mean_in_cloud = float(tile_clean.optical[:, cloud_px].mean())
    cloud_mean_in_cloud = float(tile_cloud.optical[:, cloud_px].mean())
    check(
        cloud_mean_in_cloud > clean_mean_in_cloud + 0.05,
        f"Optical is brighter in cloud region "
        f"({cloud_mean_in_cloud:.3f} vs {clean_mean_in_cloud:.3f} clean)",
    )
else:
    check(False, "Cloud mask is non-empty", "n_cloud_px == 0 — cloud generation failed")


# ── Section 4: Canopy occlusion ───────────────────────────────────────────────
print("\n[Section 4: Canopy occlusion — optical shifts to vegetation signature]")

tile_canopy = gen.generate(seed=77, occlusion_type='canopy', occlusion_fraction=0.40)
canopy_px   = tile_canopy.cloud_mask.astype(bool)
n_canopy_px = canopy_px.sum()

check(
    n_canopy_px > 50,
    f"Canopy mask has > 50 pixels ({n_canopy_px:,} canopy pixels)",
)

if n_canopy_px > 0:
    # Canopy pixels should have higher NIR (vegetation signature)
    nir_canopy_region = float(tile_canopy.optical[2][canopy_px].mean())
    check(
        nir_canopy_region > 0.25,
        f"Canopy region has elevated NIR (>{0.25:.2f}, got {nir_canopy_region:.3f})",
        "Vegetation NIR should be > 0.25 (from SPECTRAL['vegetation'][2] = 0.55)"
    )

    # SAR should be attenuated (not zero, not identical to clean)
    tile_clean2 = gen.generate(seed=77, occlusion_type='none', occlusion_fraction=0.0)
    sar_ratio   = float(tile_canopy.sar[0][canopy_px].mean()) / \
                  max(float(tile_clean2.sar[0][canopy_px].mean()), 1e-8)
    check(
        0.5 < sar_ratio < 0.95,
        f"SAR is partially attenuated under canopy (ratio={sar_ratio:.2f}, expect 0.5-0.95)",
        "SAR should be reduced but not zeroed under tree canopy"
    )


# ── Section 5: Shadow occlusion ───────────────────────────────────────────────
print("\n[Section 5: Shadow occlusion — optical darkened, SAR unchanged]")

tile_shadow = gen.generate(seed=33, occlusion_type='shadow', occlusion_fraction=0.5)
shadow_px   = tile_shadow.cloud_mask.astype(bool)

check(
    shadow_px.sum() > 0,
    f"Shadow mask is non-empty ({shadow_px.sum():,} shadow pixels)",
)

if shadow_px.sum() > 0:
    tile_clean3 = gen.generate(seed=33, occlusion_type='none', occlusion_fraction=0.0)
    shadow_opt_mean  = float(tile_shadow.optical[:, shadow_px].mean())
    clean_opt_mean   = float(tile_clean3.optical[:, shadow_px].mean())
    check(
        shadow_opt_mean < clean_opt_mean * 0.6,
        f"Shadow darkens optical (shadow={shadow_opt_mean:.3f}, "
        f"clean={clean_opt_mean:.3f}, ratio={shadow_opt_mean/max(clean_opt_mean,1e-8):.2f})",
    )

    sar_shadow_diff = float(np.abs(tile_clean3.sar - tile_shadow.sar).max())
    check(
        sar_shadow_diff < 1e-5,
        f"SAR unchanged by building shadows (max diff = {sar_shadow_diff:.2e})",
    )


# ── Section 6: Determinism ───────────────────────────────────────────────────
print("\n[Section 6: Determinism — same seed → identical tiles]")

tile_a = gen.generate(seed=123, occlusion_type='cloud', occlusion_fraction=0.35)
tile_b = gen.generate(seed=123, occlusion_type='cloud', occlusion_fraction=0.35)

check(
    np.array_equal(tile_a.optical, tile_b.optical),
    "Optical is identical for same seed",
)
check(
    np.array_equal(tile_a.sar, tile_b.sar),
    "SAR is identical for same seed",
)
check(
    np.array_equal(tile_a.gt_mask, tile_b.gt_mask),
    "GT mask is identical for same seed (road layout is fixed)",
)

tile_c = gen.generate(seed=999, occlusion_type='cloud', occlusion_fraction=0.35)
check(
    not np.array_equal(tile_a.optical, tile_c.optical),
    "Different seeds produce different optical textures",
)


# ── Section 7: Contract validation ───────────────────────────────────────────
print("\n[Section 7: Contract — road_mask.npy + meta.json pass output_writer]")

tile_demo = gen.generate(seed=42, occlusion_type='cloud', occlusion_fraction=0.4)
save_demo_output(tile_demo, gen)

try:
    mask, meta = load_and_verify(ROAD_MASK_PATH, META_PATH)
    check(True, "load_and_verify passes on demo output")
    check(
        meta['crs'] == "EPSG:4326",
        f"meta.crs == 'EPSG:4326' (got '{meta['crs']}')"
    )
    check(
        meta['source'] == "synthetic",
        f"meta.source == 'synthetic' (got '{meta['source']}')"
    )
    check(
        mask.dtype == np.uint8,
        f"Loaded mask dtype is uint8 (got {mask.dtype})"
    )
    check(
        set(np.unique(mask).tolist()).issubset({0, 1}),
        "Loaded mask values in {0, 1}",
    )
except Exception as e:
    check(False, "load_and_verify passes on demo output", str(e))


# ── Section 8: Visualization PNG ─────────────────────────────────────────────
print("\n[Section 8: Visualization PNG saved and non-empty]")

import tempfile
tmp_dir  = Path(tempfile.mkdtemp())
vis_path = tmp_dir / "test_visualization.png"
visualize_tile(tile_demo, str(vis_path), title="Test Visualization")

check(
    vis_path.exists(),
    "Visualization PNG was created",
    f"Expected at: {vis_path}"
)
if vis_path.exists():
    size_kb = vis_path.stat().st_size / 1024
    check(
        size_kb > 50,
        f"Visualization PNG is non-trivial ({size_kb:.1f} KB, expect >50 KB)",
    )


# ── Section 9: Dataset generation ────────────────────────────────────────────
print("\n[Section 9: Dataset generation — 5 train + 2 val tiles]")

dataset_dir = str(tmp_dir / "dataset_test")
result = generate_dataset(
    generator  = gen,
    n_train    = 5,
    n_val      = 2,
    output_dir = dataset_dir,
    verbose    = False,
)

train_dir = Path(result['train_dir'])
val_dir   = Path(result['val_dir'])

check(
    train_dir.exists(),
    f"Train directory created: {train_dir.name}"
)
check(
    val_dir.exists(),
    f"Val directory created: {val_dir.name}"
)

train_npz_files = list(train_dir.glob("*.npz"))
check(
    len(train_npz_files) == 5,
    f"5 training tiles saved as .npz ({len(train_npz_files)} found)"
)

# Load and verify one tile
if train_npz_files:
    data = np.load(str(train_npz_files[0]))
    check(
        'optical' in data and 'sar' in data and 'gt_mask' in data,
        "Loaded .npz contains optical, sar, gt_mask keys"
    )
    check(
        data['optical'].shape == (4, 512, 512),
        f"Loaded optical shape is (4, 512, 512), got {data['optical'].shape}"
    )

val_npz_files = list(val_dir.glob("*.npz"))
check(
    len(val_npz_files) == 2,
    f"2 validation tiles saved as .npz ({len(val_npz_files)} found)"
)


# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 60)
total = _pass + _fail
print(f"  Results: {_pass} / {total} passed   ({_fail} failed)")
print()
if _fail == 0:
    print("  ✓ ALL PHASE 2 TESTS PASSED")
    print("  ✓ Phase 2 exit criterion: MET")
    print()
    print("  Phase 2 outputs:")
    print(f"    part_a_vision/outputs/road_mask.npy       ← GT mask, uint8")
    print(f"    part_a_vision/outputs/meta.json           ← CRS, bbox, source")
    print(f"    part_a_vision/outputs/demo_visuals/       ← visualizations")
    print(f"    part_a_vision/data/koramangala/           ← cached OSMnx + GT")
    print()
    print("  → Ready for Phase 3: Evaluation Harness (eval.py)")
    print("  → Part B can load road_mask.npy NOW for integration testing")
else:
    print(f"  ✗ {_fail} TEST(S) FAILED — fix before Phase 3")
print("=" * 60)
print()
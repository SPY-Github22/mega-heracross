# part_a_vision/output_writer.py
# Contract-enforcing output writer for Part A.
# This module is the ONLY place that writes road_mask.npy and meta.json.
# It validates the contract on every write and re-validates on every load.
# If the contract is violated, it raises a clear ValueError - never silently corrupts.

import numpy as np
import json
import os
from pathlib import Path
from dataclasses import asdict
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_here)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from shared.schema import RoadMaskMeta
from shared.config import TARGET_CRS, ROAD_MASK_PATH, META_PATH


# ── Internal Validators ────────────────────────────────────────────────────────
# These are called by write_* and load_and_verify both.
# Separated from save logic so they can be reused without side effects.

def _validate_mask(mask_array: np.ndarray) -> None:
    """
    Raises ValueError with a clear message if mask_array violates the contract.
    Contract: np.ndarray, dtype=uint8, shape=(H, W), values in {0, 1}.
    """
    if not isinstance(mask_array, np.ndarray):
        raise ValueError(
            f"[output_writer] road_mask must be np.ndarray.\n"
            f"  Got: {type(mask_array)}\n"
            f"  Fix: Pass a numpy array."
        )

    if mask_array.dtype != np.uint8:
        raise ValueError(
            f"[output_writer] road_mask dtype must be uint8.\n"
            f"  Got: {mask_array.dtype}\n"
            f"  Fix: After thresholding, cast with: mask = (prob_map > 0.5).astype(np.uint8)"
        )

    if mask_array.ndim != 2:
        raise ValueError(
            f"[output_writer] road_mask must be 2D (H, W).\n"
            f"  Got shape: {mask_array.shape} ({mask_array.ndim}D)\n"
            f"  Fix: If shape is (1, H, W), squeeze with: mask = mask.squeeze(0)"
        )

    if mask_array.shape[0] == 0 or mask_array.shape[1] == 0:
        raise ValueError(
            f"[output_writer] road_mask has a zero-length dimension.\n"
            f"  Got shape: {mask_array.shape}"
        )

    unique_vals = set(np.unique(mask_array).tolist())
    if not unique_vals.issubset({0, 1}):
        bad_vals = unique_vals - {0, 1}
        raise ValueError(
            f"[output_writer] road_mask values must be exactly 0 or 1.\n"
            f"  Got unexpected values: {bad_vals}\n"
            f"  Fix: If values are 0 and 255 (from cv2), use: mask = (mask // 255).astype(np.uint8)\n"
            f"  Fix: If values are floats, use: mask = (prob_map > 0.5).astype(np.uint8)"
        )


def _validate_meta(crs: str, bbox, resolution_m, source: str) -> None:
    """
    Raises ValueError with a clear message if any meta field violates the contract.
    """
    if crs != TARGET_CRS:
        raise ValueError(
            f"[output_writer] meta.crs must be '{TARGET_CRS}'.\n"
            f"  Got: '{crs}'\n"
            f"  Fix: Always reproject tiles to EPSG:4326 before processing."
        )

    if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
        raise ValueError(
            f"[output_writer] meta.bbox must be a 4-element tuple: (min_lon, min_lat, max_lon, max_lat).\n"
            f"  Got: {bbox}"
        )

    min_lon, min_lat, max_lon, max_lat = bbox

    if not isinstance(min_lon, (int, float)):
        raise ValueError(f"[output_writer] bbox values must be numeric. Got min_lon={min_lon!r}")

    if not (min_lon < max_lon):
        raise ValueError(
            f"[output_writer] meta.bbox: min_lon must be < max_lon.\n"
            f"  Got: min_lon={min_lon}, max_lon={max_lon}\n"
            f"  Fix: Check bbox order - it must be (min_lon, min_lat, max_lon, max_lat)"
        )

    if not (min_lat < max_lat):
        raise ValueError(
            f"[output_writer] meta.bbox: min_lat must be < max_lat.\n"
            f"  Got: min_lat={min_lat}, max_lat={max_lat}"
        )

    # Sanity check: values should be within plausible geographic ranges
    if not (-180 <= min_lon <= 180) or not (-180 <= max_lon <= 180):
        raise ValueError(f"[output_writer] bbox longitude values out of range [-180, 180]: {min_lon}, {max_lon}")
    if not (-90 <= min_lat <= 90) or not (-90 <= max_lat <= 90):
        raise ValueError(f"[output_writer] bbox latitude values out of range [-90, 90]: {min_lat}, {max_lat}")

    if not isinstance(resolution_m, (int, float)) or resolution_m <= 0:
        raise ValueError(
            f"[output_writer] meta.resolution_m must be a positive number.\n"
            f"  Got: {resolution_m!r}"
        )

    if not isinstance(source, str) or len(source.strip()) == 0:
        raise ValueError(
            f"[output_writer] meta.source must be a non-empty string.\n"
            f"  Got: {source!r}\n"
            f"  Examples: 'LISS-IV', 'Sentinel-2', 'synthetic'"
        )


# ── Public API ────────────────────────────────────────────────────────────────

def write_road_mask(mask_array: np.ndarray, path: str = ROAD_MASK_PATH) -> None:
    """
    Validate and save road_mask.npy.

    Args:
        mask_array: np.ndarray, dtype=uint8, shape=(H, W), values in {0, 1}
        path: output path (default: from shared/config.py)

    Raises:
        ValueError: if any contract constraint is violated (with fix hint)
    """
    _validate_mask(mask_array)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, mask_array)

    road_px  = int(mask_array.sum())
    total_px = int(mask_array.size)
    frac     = road_px / total_px * 100

    print(f"[OK] road_mask.npy saved")
    print(f"  Path:         {path}")
    print(f"  Shape:        {mask_array.shape} (H={mask_array.shape[0]}, W={mask_array.shape[1]})")
    print(f"  Road pixels:  {road_px:,} / {total_px:,} ({frac:.1f}%)")


def write_meta(
    crs:          str,
    bbox:         tuple,
    resolution_m: float,
    source:       str,
    path:         str = META_PATH
) -> None:
    """
    Validate and save meta.json.

    Args:
        crs:          coordinate reference system - must be "EPSG:4326"
        bbox:         (min_lon, min_lat, max_lon, max_lat) in EPSG:4326
        resolution_m: spatial resolution in metres per pixel
        source:       data source description
        path:         output path (default: from shared/config.py)

    Raises:
        ValueError: if any contract constraint is violated (with fix hint)
    """
    _validate_meta(crs, bbox, resolution_m, source)

    meta = RoadMaskMeta(
        crs=crs,
        bbox=tuple(float(v) for v in bbox),
        resolution_m=float(resolution_m),
        source=source
    )

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(asdict(meta), f, indent=2)

    print(f"[OK] meta.json saved")
    print(f"  Path:         {path}")
    print(f"  CRS:          {crs}")
    print(f"  BBox:         {bbox}")
    print(f"  Resolution:   {resolution_m}m/pixel")
    print(f"  Source:       {source}")


def load_and_verify(
    mask_path: str = ROAD_MASK_PATH,
    meta_path: str = META_PATH
) -> tuple:
    """
    Load road_mask.npy and meta.json, re-validate both against contract.
    Call this at the end of every pipeline run.

    Returns:
        (mask_array, meta_dict) if both pass validation.

    Raises:
        FileNotFoundError: if files don't exist
        ValueError: if contract is violated
    """
    print(f"\n[load_and_verify] Checking Part A outputs...")

    # Load mask
    if not os.path.exists(mask_path):
        raise FileNotFoundError(
            f"[load_and_verify] road_mask.npy not found at '{mask_path}'.\n"
            f"  Fix: Run the Part A pipeline first, or check ROAD_MASK_PATH in shared/config.py"
        )
    mask = np.load(mask_path)
    _validate_mask(mask)

    # Load meta
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"[load_and_verify] meta.json not found at '{meta_path}'.\n"
            f"  Fix: Run the Part A pipeline first, or check META_PATH in shared/config.py"
        )
    with open(meta_path, 'r') as f:
        meta_dict = json.load(f)

    # Check all required fields exist before validating their values
    required_fields = {'crs', 'bbox', 'resolution_m', 'source'}
    missing = required_fields - set(meta_dict.keys())
    if missing:
        raise ValueError(
            f"[load_and_verify] meta.json is missing required fields: {missing}\n"
            f"  Present fields: {set(meta_dict.keys())}"
        )

    _validate_meta(
        meta_dict['crs'],
        meta_dict['bbox'],
        meta_dict['resolution_m'],
        meta_dict['source']
    )

    road_px  = int(mask.sum())
    total_px = int(mask.size)

    print(f"[OK] Contract validation PASSED")
    print(f"  Mask:   {mask.shape}, dtype={mask.dtype}, "
          f"road={road_px:,}/{total_px:,} ({road_px/total_px*100:.1f}%)")
    print(f"  Meta:   crs={meta_dict['crs']}, source={meta_dict['source']}, "
          f"resolution={meta_dict['resolution_m']}m")

    return mask, meta_dict
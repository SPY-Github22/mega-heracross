"""
part_b_skeleton/loader.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 03: Synthetic mask loader + geo-transform

Responsibilities:
  1. Load road_mask.npy → np.ndarray (dtype=uint8, values 0/1)
  2. Load meta.json     → RoadMaskMeta dataclass
  3. Build pixel→lat/lon affine transform from bbox + mask shape
  4. Expose pixel_to_latlon(row, col) and latlon_to_pixel(lat, lon)
  5. Validate everything — fail loud, fail early

The geo-transform:
  bbox = (min_lon, min_lat, max_lon, max_lat)   ← from meta.json
  Pixel (row=0, col=0)   → (lat=max_lat, lon=min_lon)  [top-left]
  Pixel (row=H, col=W)   → (lat=min_lat, lon=max_lon)  [bottom-right]

  lon(col) = min_lon + col * (max_lon - min_lon) / W
  lat(row) = max_lat - row * (max_lat - min_lat) / H

  Note: row increases DOWNWARD, lat increases UPWARD — hence the minus sign.
  This is the standard raster convention (same as GDAL, rasterio, numpy).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
import os
import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

# ── repo-root-relative imports ────────────────────────────────
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from shared.schema import RoadMaskMeta
from shared.config import TARGET_CRS


# ══════════════════════════════════════════════════════════════
# AFFINE TRANSFORM
# ══════════════════════════════════════════════════════════════

@dataclass
class AffineTransform:
    """
    Lightweight affine transform for pixel ↔ lat/lon conversion.
    Derived entirely from bbox + mask shape — no external library needed.

    Attributes
    ----------
    min_lon, min_lat, max_lon, max_lat : float
        Bounding box in EPSG:4326.
    width, height : int
        Mask dimensions in pixels (W columns, H rows).
    lon_per_pixel : float
        Degrees longitude per pixel column.
    lat_per_pixel : float
        Degrees latitude per pixel row (positive value; applied as subtraction).
    resolution_m : float
        Ground sampling distance in metres (from meta.json).
    """
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float
    width: int      # W — number of columns
    height: int     # H — number of rows
    lon_per_pixel: float
    lat_per_pixel: float
    resolution_m: float

    def pixel_to_latlon(self, row: float, col: float) -> Tuple[float, float]:
        """
        Convert pixel (row, col) to (lat, lon) in EPSG:4326.

        Parameters
        ----------
        row : float  — pixel row index (0 = top of image)
        col : float  — pixel column index (0 = left of image)

        Returns
        -------
        (lat, lon) : Tuple[float, float]
        """
        lon = self.min_lon + col * self.lon_per_pixel
        lat = self.max_lat - row * self.lat_per_pixel
        return (lat, lon)

    def latlon_to_pixel(self, lat: float, lon: float) -> Tuple[float, float]:
        """
        Convert (lat, lon) in EPSG:4326 to pixel (row, col).
        Inverse of pixel_to_latlon.

        Returns
        -------
        (row, col) : Tuple[float, float]  — may be fractional
        """
        col = (lon - self.min_lon) / self.lon_per_pixel
        row = (self.max_lat - lat) / self.lat_per_pixel
        return (row, col)

    def pixel_distance_to_metres(self, n_pixels: float) -> float:
        """
        Approximate real-world distance for n_pixels along one axis.
        Uses resolution_m from meta.json (ground truth from Part A).
        """
        return n_pixels * self.resolution_m

    def __repr__(self) -> str:
        return (
            f"AffineTransform("
            f"bbox=({self.min_lon:.4f},{self.min_lat:.4f},"
            f"{self.max_lon:.4f},{self.max_lat:.4f}), "
            f"size={self.width}×{self.height}px, "
            f"res={self.resolution_m:.1f}m/px)"
        )


# ══════════════════════════════════════════════════════════════
# META.JSON LOADER
# ══════════════════════════════════════════════════════════════

def load_meta(meta_path: str) -> RoadMaskMeta:
    """
    Load and validate meta.json → RoadMaskMeta.

    Checks:
      • File exists and is valid JSON
      • All required fields present: crs, bbox, resolution_m, source
      • crs == 'EPSG:4326'
      • bbox is (min_lon, min_lat, max_lon, max_lat) with min < max
      • resolution_m > 0
      • source is a non-empty string

    Raises
    ------
    FileNotFoundError  — if meta.json doesn't exist
    ValueError         — if any field is invalid
    """
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"meta.json not found at: {meta_path}\n"
            f"Part A must run first and emit this file."
        )

    with open(meta_path, "r") as f:
        try:
            raw = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"meta.json is not valid JSON: {e}")

    # ── Required fields ───────────────────────────────────────
    required = {"crs", "bbox", "resolution_m", "source"}
    missing = required - set(raw.keys())
    if missing:
        raise ValueError(f"meta.json missing required fields: {sorted(missing)}")

    # ── CRS ───────────────────────────────────────────────────
    crs = raw["crs"]
    if crs != TARGET_CRS:
        raise ValueError(
            f"meta.json crs='{crs}' but must be '{TARGET_CRS}' — "
            f"CRS is a locked constant. Part A must reproject before emitting."
        )

    # ── bbox ──────────────────────────────────────────────────
    bbox = raw["bbox"]
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(
            f"meta.json bbox must be a 4-element list [min_lon, min_lat, max_lon, max_lat], "
            f"got: {bbox!r}"
        )
    min_lon, min_lat, max_lon, max_lat = [float(x) for x in bbox]

    if min_lon >= max_lon:
        raise ValueError(f"bbox: min_lon ({min_lon}) must be < max_lon ({max_lon})")
    if min_lat >= max_lat:
        raise ValueError(f"bbox: min_lat ({min_lat}) must be < max_lat ({max_lat})")

    # Sanity: bbox should be in India roughly
    if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180):
        raise ValueError(f"bbox longitudes out of [-180,180] range: {min_lon}, {max_lon}")
    if not (-90 <= min_lat <= 90 and -90 <= max_lat <= 90):
        raise ValueError(f"bbox latitudes out of [-90,90] range: {min_lat}, {max_lat}")

    # ── resolution_m ──────────────────────────────────────────
    resolution_m = float(raw["resolution_m"])
    if resolution_m <= 0:
        raise ValueError(f"resolution_m must be > 0, got {resolution_m}")
    if resolution_m > 1000:
        raise ValueError(
            f"resolution_m={resolution_m}m seems unreasonably large — "
            f"LISS-IV is 5.8m, Sentinel-2 is 10m"
        )

    # ── source ────────────────────────────────────────────────
    source = str(raw["source"]).strip()
    if not source:
        raise ValueError("meta.json 'source' field is empty")

    return RoadMaskMeta(
        crs=crs,
        bbox=(min_lon, min_lat, max_lon, max_lat),
        resolution_m=resolution_m,
        source=source,
    )


# ══════════════════════════════════════════════════════════════
# MASK LOADER
# ══════════════════════════════════════════════════════════════

def load_mask(mask_path: str) -> np.ndarray:
    """
    Load road_mask.npy → np.ndarray.

    Contract (from shared/schema.py):
      dtype  : uint8
      shape  : (H, W)  — 2D only
      values : 0 or 1  — binary road mask

    Raises
    ------
    FileNotFoundError  — if road_mask.npy doesn't exist
    ValueError         — if array doesn't meet contract
    """
    if not os.path.exists(mask_path):
        raise FileNotFoundError(
            f"road_mask.npy not found at: {mask_path}\n"
            f"Part A must run first and emit this file."
        )

    mask = np.load(mask_path)

    # ── dtype ─────────────────────────────────────────────────
    if mask.dtype != np.uint8:
        # Be helpful — try to auto-convert if values are 0/1
        unique_vals = np.unique(mask)
        if set(unique_vals.tolist()).issubset({0, 1}):
            mask = mask.astype(np.uint8)
        else:
            raise ValueError(
                f"road_mask.npy dtype={mask.dtype} — contract requires uint8. "
                f"Unique values found: {unique_vals[:10]}"
            )

    # ── shape ─────────────────────────────────────────────────
    if mask.ndim != 2:
        raise ValueError(
            f"road_mask.npy shape={mask.shape} — contract requires 2D (H, W). "
            f"Got {mask.ndim} dimensions."
        )

    H, W = mask.shape
    if H < 10 or W < 10:
        raise ValueError(
            f"road_mask.npy shape={mask.shape} is suspiciously small — "
            f"minimum expected is 10×10 pixels."
        )

    # ── values ────────────────────────────────────────────────
    unique_vals = set(np.unique(mask).tolist())
    invalid_vals = unique_vals - {0, 1}
    if invalid_vals:
        raise ValueError(
            f"road_mask.npy contains values other than 0 and 1: {invalid_vals}. "
            f"Contract requires binary mask."
        )

    road_pixels = int(mask.sum())
    total_pixels = H * W
    road_fraction = road_pixels / total_pixels

    if road_pixels == 0:
        raise ValueError(
            "road_mask.npy has zero road pixels — mask appears empty. "
            "Check Part A segmentation output."
        )
    if road_fraction > 0.8:
        raise ValueError(
            f"road_mask.npy is {road_fraction:.1%} road pixels — this is unrealistically high. "
            f"Typical urban road coverage is 5–30%."
        )

    return mask


# ══════════════════════════════════════════════════════════════
# AFFINE TRANSFORM BUILDER
# ══════════════════════════════════════════════════════════════

def build_affine(mask: np.ndarray, meta: RoadMaskMeta) -> AffineTransform:
    """
    Build the pixel→lat/lon AffineTransform from mask shape + meta bbox.

    The transform is purely derived from the bbox and mask dimensions:
      lon(col) = min_lon + col * (max_lon - min_lon) / W
      lat(row) = max_lat - row * (max_lat - min_lat) / H

    Also validates that resolution_m in meta.json is consistent with
    the bbox extent and mask dimensions (within 20% tolerance).

    Parameters
    ----------
    mask : np.ndarray  — shape (H, W)
    meta : RoadMaskMeta

    Returns
    -------
    AffineTransform
    """
    H, W = mask.shape
    min_lon, min_lat, max_lon, max_lat = meta.bbox

    lon_per_pixel = (max_lon - min_lon) / W
    lat_per_pixel = (max_lat - min_lat) / H

    # ── Validate resolution_m consistency ─────────────────────
    # Estimate metres per pixel from bbox extent using Haversine
    # (centre latitude for Koramangala ≈ 12.9277°N)
    centre_lat = (min_lat + max_lat) / 2.0
    lat_rad = math.radians(centre_lat)

    # 1 degree latitude ≈ 111,320 m (nearly constant)
    # 1 degree longitude ≈ 111,320 * cos(lat) m
    metres_per_deg_lat = 111_320.0
    metres_per_deg_lon = 111_320.0 * math.cos(lat_rad)

    # Pixel size in metres (average of lat and lon directions)
    pixel_size_from_lat = (lat_per_pixel * metres_per_deg_lat)
    pixel_size_from_lon = (lon_per_pixel * metres_per_deg_lon)
    computed_res_m = (pixel_size_from_lat + pixel_size_from_lon) / 2.0

    # Tolerance check: computed vs declared resolution_m
    ratio = computed_res_m / meta.resolution_m
    if not (0.5 <= ratio <= 2.0):
        raise ValueError(
            f"resolution_m mismatch: meta.json says {meta.resolution_m:.1f}m/px but "
            f"bbox+shape implies {computed_res_m:.1f}m/px (ratio={ratio:.2f}). "
            f"Check that bbox and mask dimensions are consistent."
        )

    return AffineTransform(
        min_lon=min_lon,
        min_lat=min_lat,
        max_lon=max_lon,
        max_lat=max_lat,
        width=W,
        height=H,
        lon_per_pixel=lon_per_pixel,
        lat_per_pixel=lat_per_pixel,
        resolution_m=meta.resolution_m,
    )


# ══════════════════════════════════════════════════════════════
# CORNER VALIDATION  (the most important test in Phase 03)
# ══════════════════════════════════════════════════════════════

def validate_corners(affine: AffineTransform, meta: RoadMaskMeta,
                     tolerance_deg: float = 1e-6) -> list:
    """
    Verify the four corners of the affine transform map exactly to
    the bbox corners from meta.json.

    This is the Phase 03 correctness proof — if corners don't round-trip,
    the geo-transform is wrong and every subsequent lat/lon is garbage.

    Returns list of violation strings (empty = all corners correct).
    """
    violations = []
    min_lon, min_lat, max_lon, max_lat = meta.bbox
    H, W = affine.height, affine.width

    expected_corners = {
        "top-left     (row=0, col=0)":   (max_lat, min_lon),
        "top-right    (row=0, col=W)":   (max_lat, max_lon),
        "bottom-left  (row=H, col=0)":   (min_lat, min_lon),
        "bottom-right (row=H, col=W)":   (min_lat, max_lon),
    }
    pixel_corners = {
        "top-left     (row=0, col=0)":   (0, 0),
        "top-right    (row=0, col=W)":   (0, W),
        "bottom-left  (row=H, col=0)":   (H, 0),
        "bottom-right (row=H, col=W)":   (H, W),
    }

    for label, (row, col) in pixel_corners.items():
        computed_lat, computed_lon = affine.pixel_to_latlon(row, col)
        expected_lat, expected_lon = expected_corners[label]

        lat_err = abs(computed_lat - expected_lat)
        lon_err = abs(computed_lon - expected_lon)

        if lat_err > tolerance_deg or lon_err > tolerance_deg:
            violations.append(
                f"{label}: computed=({computed_lat:.8f}, {computed_lon:.8f}) "
                f"expected=({expected_lat:.8f}, {expected_lon:.8f}) "
                f"error=(Δlat={lat_err:.2e}, Δlon={lon_err:.2e})"
            )

    return violations


# ══════════════════════════════════════════════════════════════
# SYNTHETIC MASK GENERATOR  (for testing when Part A hasn't run)
# ══════════════════════════════════════════════════════════════

def make_synthetic_koramangala_mask(height: int = 200,
                                    width: int = 200) -> Tuple[np.ndarray, RoadMaskMeta]:
    """
    Generate a synthetic road mask for the Koramangala test tile.
    Used when Part A's road_mask.npy is not yet available.

    The synthetic mask draws a realistic grid of roads:
      • 3 horizontal arterials (evenly spaced)
      • 3 vertical arterials (evenly spaced)
      • Road width: 3 pixels (representing ~17m at 5.8m/px)
      • 2 deliberate occlusion gaps (simulating tree canopy)

    Returns
    -------
    (mask, meta) : (np.ndarray uint8 shape HxW, RoadMaskMeta)
    """
    from shared.config import TEST_TILE_BBOX

    mask = np.zeros((height, width), dtype=np.uint8)
    road_width = 3

    # ── Horizontal arterials ──────────────────────────────────
    h_positions = [height // 4, height // 2, 3 * height // 4]
    for r in h_positions:
        for dr in range(-road_width // 2, road_width // 2 + 1):
            row = r + dr
            if 0 <= row < height:
                mask[row, :] = 1

    # ── Vertical arterials ────────────────────────────────────
    v_positions = [width // 4, width // 2, 3 * width // 4]
    for c in v_positions:
        for dc in range(-road_width // 2, road_width // 2 + 1):
            col = c + dc
            if 0 <= col < width:
                mask[:, col] = 1

    # ── Occlusion gaps (simulating tree canopy / cloud) ───────
    # Gap 1: middle of top horizontal road
    gap1_row = h_positions[0]
    gap1_col_start, gap1_col_end = width // 3, width // 3 + 15
    for dr in range(-road_width // 2, road_width // 2 + 1):
        row = gap1_row + dr
        if 0 <= row < height:
            mask[row, gap1_col_start:gap1_col_end] = 0

    # Gap 2: middle of right vertical road
    gap2_col = v_positions[2]
    gap2_row_start, gap2_row_end = height // 3, height // 3 + 12
    for dc in range(-road_width // 2, road_width // 2 + 1):
        col = gap2_col + dc
        if 0 <= col < width:
            mask[gap2_row_start:gap2_row_end, col] = 0

    # ── Compute resolution_m from bbox ────────────────────────
    min_lon, min_lat, max_lon, max_lat = TEST_TILE_BBOX
    centre_lat = (min_lat + max_lat) / 2.0
    metres_per_deg_lat = 111_320.0
    metres_per_deg_lon = 111_320.0 * math.cos(math.radians(centre_lat))
    bbox_height_m = (max_lat - min_lat) * metres_per_deg_lat
    bbox_width_m  = (max_lon - min_lon) * metres_per_deg_lon
    resolution_m  = (bbox_height_m / height + bbox_width_m / width) / 2.0

    meta = RoadMaskMeta(
        crs="EPSG:4326",
        bbox=TEST_TILE_BBOX,
        resolution_m=round(resolution_m, 2),
        source="synthetic",
    )

    return mask, meta


# ══════════════════════════════════════════════════════════════
# TOP-LEVEL LOAD FUNCTION  (called by run.py)
# ══════════════════════════════════════════════════════════════

def load_inputs(mask_path: str, meta_path: str,
                use_synthetic_fallback: bool = True
                ) -> Tuple[np.ndarray, RoadMaskMeta, AffineTransform]:
    """
    Master loader: load mask + meta, build affine transform.
    If Part A files don't exist and use_synthetic_fallback=True,
    generates a synthetic Koramangala mask instead.

    Returns
    -------
    (mask, meta, affine) : Tuple[np.ndarray, RoadMaskMeta, AffineTransform]
    """
    part_a_exists = os.path.exists(mask_path) and os.path.exists(meta_path)

    if not part_a_exists:
        if use_synthetic_fallback:
            print(f"  ○ Part A outputs not found — using synthetic Koramangala mask")
            mask, meta = make_synthetic_koramangala_mask()
        else:
            raise FileNotFoundError(
                f"Part A outputs missing:\n"
                f"  mask: {mask_path}\n"
                f"  meta: {meta_path}\n"
                f"Run Part A first, or set use_synthetic_fallback=True."
            )
    else:
        print(f"  ✓ Loading Part A mask: {mask_path}")
        print(f"  ✓ Loading Part A meta: {meta_path}")
        mask = load_mask(mask_path)
        meta = load_meta(meta_path)

    affine = build_affine(mask, meta)
    return mask, meta, affine


def print_loader_report(mask: np.ndarray, meta: RoadMaskMeta,
                        affine: AffineTransform) -> dict:
    """
    Print Phase 03 loader report and return metrics dict.
    """
    SEP = "─" * 60
    H, W = mask.shape
    road_pixels = int(mask.sum())
    road_fraction = road_pixels / (H * W)

    # Corner validation
    corner_violations = validate_corners(affine, meta)

    # Centre pixel round-trip test
    centre_row, centre_col = H / 2, W / 2
    centre_lat, centre_lon = affine.pixel_to_latlon(centre_row, centre_col)
    rt_row, rt_col = affine.latlon_to_pixel(centre_lat, centre_lon)
    roundtrip_err_px = math.sqrt((rt_row - centre_row)**2 + (rt_col - centre_col)**2)

    print(f"\n{SEP}")
    print(f"  PHASE 03 — MASK LOADER & GEO-TRANSFORM")
    print(SEP)
    print(f"  Source      : {meta.source}")
    print(f"  CRS         : {meta.crs}")
    print(f"  Mask shape  : {H} × {W} pixels")
    print(f"  Road pixels : {road_pixels:,} / {H*W:,} ({road_fraction:.1%})")
    print(f"  Resolution  : {meta.resolution_m:.2f} m/px")
    print(f"  BBox        : lon [{meta.bbox[0]:.4f}, {meta.bbox[2]:.4f}]")
    print(f"               lat [{meta.bbox[1]:.4f}, {meta.bbox[3]:.4f}]")
    print(f"\n  Geo-transform: {affine}")
    print(f"    lon_per_pixel = {affine.lon_per_pixel:.8f}°")
    print(f"    lat_per_pixel = {affine.lat_per_pixel:.8f}°")
    print(f"\n  Corner validation:")

    if corner_violations:
        for v in corner_violations:
            print(f"    ✗ {v}")
    else:
        min_lon, min_lat, max_lon, max_lat = meta.bbox
        tl_lat, tl_lon = affine.pixel_to_latlon(0, 0)
        br_lat, br_lon = affine.pixel_to_latlon(H, W)
        print(f"    ✓ top-left  (0,0)   → lat={tl_lat:.6f}, lon={tl_lon:.6f}")
        print(f"    ✓ bot-right ({H},{W}) → lat={br_lat:.6f}, lon={br_lon:.6f}")
        print(f"    ✓ All 4 corners within 1e-6° of bbox")

    print(f"\n  Round-trip test (pixel→latlon→pixel):")
    print(f"    centre pixel ({centre_row:.1f}, {centre_col:.1f})")
    print(f"    → latlon ({centre_lat:.6f}, {centre_lon:.6f})")
    print(f"    → pixel  ({rt_row:.6f}, {rt_col:.6f})")
    print(f"    error = {roundtrip_err_px:.2e} pixels")
    rt_ok = roundtrip_err_px < 1e-6
    print(f"    {'✓ PASS' if rt_ok else '✗ FAIL'} (threshold: 1e-6 px)")

    overall_ok = not corner_violations and rt_ok
    print(f"\n{SEP}")
    print(f"  LOADER: {'✓ PASS' if overall_ok else '✗ FAIL'}")
    print(SEP)

    return {
        "source": meta.source,
        "mask_shape": (H, W),
        "road_pixels": road_pixels,
        "road_fraction": road_fraction,
        "resolution_m": meta.resolution_m,
        "corner_violations": corner_violations,
        "roundtrip_error_px": roundtrip_err_px,
        "loader_pass": overall_ok,
    }

#!/usr/bin/env python3
"""
Phase 5 — SAR Preprocessing Pipeline
=====================================
Sentinel-1 GRD ingestion: VV/VH loading, dB conversion, Lee speckle filter,
percentile normalization, upsampling to optical resolution, and a synthetic
SAR generator for fallback when real data is unavailable.

Place this file at: part_a_vision/sar_reader.py

Exit Criterion:
    sar_reader.py produces a (2, H, W) float32 normalized array with visible
    Lee-filtered SAR texture, roads showing as dark linear features in VV.

Usage:
    from part_a_vision.sar_reader import SARPreprocessor
    preprocessor = SARPreprocessor()
    sar_tensor, meta = preprocessor.process("path/to/vv.tif", "path/to/vh.tif")
    # sar_tensor.shape → (2, H, W) float32, normalized
    # meta['road_darkness_vv_mean'] → ~ -14 dB typical

Key Physics (for judge presentations):
    Roads appear DARK in SAR because smooth asphalt acts as a specular
    reflector — the radar pulse bounces away from the antenna (forward
    scatter), returning almost no signal.  Buildings appear BRIGHT due
    to double-bounce (ground + wall → back to antenna) and corner
    reflections.  This is why SAR complements optical: clouds block
    optics but are transparent to C-band radar.

Caveats:
    - Sentinel-1 at 10 m is coarser than LISS-IV at 5.8 m.  Upsampling
      to 5.8 m does NOT add information — it simply aligns grids for
      Phase 6 fusion.  Flag this honestly.
    - Lee filter is a despeckling heuristic, not a learned denoiser.
      Phase 13 may replace it with a learned variant.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger("sar_reader")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(_h)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default bounding box for Bengaluru test tile (EPSG:4326) — Koramangala area
DEFAULT_BBOX: Tuple[float, float, float, float] = (
    77.6050,  # min_lon (west)
    12.9250,  # min_lat (south)
    77.6300,  # max_lon (east)
    12.9450,  # max_lat (north)
)

# SAR resolution defaults
SAR_NATIVE_RESOLUTION_M: float = 10.0       # Sentinel-1 GRD IW
OPTICAL_TARGET_RESOLUTION_M: float = 5.8    # LISS-IV
TARGET_CRS: str = "EPSG:4326"

# dB conversion
DB_EPSILON: float = 1e-10
TYPICAL_URBAN_DB_RANGE: Tuple[float, float] = (-20.0, 5.0)
TYPICAL_ROAD_DB_RANGE: Tuple[float, float] = (-15.0, -12.0)

# Lee filter defaults
LEE_WINDOW_SIZE: int = 7
LEE_NOISE_VARIANCE: float = 0.273  # Equivalent number of looks ≈ 1 for single-look

# Normalisation percentiles
PERCENTILE_LOW: float = 1.0
PERCENTILE_HIGH: float = 99.0


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SARMeta:
    """Metadata collected during SAR preprocessing for reproducibility."""
    source_path_vv: Optional[str] = None
    source_path_vh: Optional[str] = None
    source_name: str = "unknown"
    source_crs: Optional[str] = None
    native_resolution_m: Optional[float] = None
    target_resolution_m: float = OPTICAL_TARGET_RESOLUTION_M
    original_shape: Optional[Tuple[int, int]] = None
    output_shape: Optional[Tuple[int, ...]] = None
    is_synthetic: bool = False
    is_db: bool = True
    lee_window: int = LEE_WINDOW_SIZE
    lee_noise_variance: float = LEE_NOISE_VARIANCE
    norm_percentiles: Tuple[float, float] = (PERCENTILE_LOW, PERCENTILE_HIGH)
    vv_db_min: Optional[float] = None
    vv_db_max: Optional[float] = None
    vh_db_min: Optional[float] = None
    vh_db_max: Optional[float] = None
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


# ---------------------------------------------------------------------------
# SARPreprocessor
# ---------------------------------------------------------------------------

class SARPreprocessor:
    """
    Full SAR ingestion and preprocessing pipeline for Sentinel-1 GRD.

    Parameters
    ----------
    bbox : tuple of float
        (min_lon, min_lat, max_lon, max_lat) for clipping.
    target_size : int
        Desired spatial dimension (square) snapped to multiple of 32.
    lee_window : int
        Lee filter window size (default 7).
    lee_noise_variance : float
        Speckle noise variance for Lee filter (default 0.273 for ENL≈1).
    """

    def __init__(
        self,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        target_size: int = 512,
        lee_window: int = LEE_WINDOW_SIZE,
        lee_noise_variance: float = LEE_NOISE_VARIANCE,
    ):
        self.bbox = bbox or DEFAULT_BBOX
        self.target_size = self._snap_to_multiple(target_size, 32)
        self.lee_window = lee_window
        self.lee_noise_variance = lee_noise_variance

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def process(
        self,
        vv_path: Optional[str] = None,
        vh_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, SARMeta]:
        """
        Run the full SAR preprocessing pipeline.

        Parameters
        ----------
        vv_path : str or None
            Path to VV polarization GeoTIFF.
        vh_path : str or None
            Path to VH polarization GeoTIFF.

        Returns
        -------
        sar_tensor : np.ndarray, shape (2, H, W), dtype float32
            Normalised SAR tensor (ch0=VV, ch1=VH).
        meta : SARMeta
            All collected metadata.
        """
        meta = SARMeta()
        meta.lee_window = self.lee_window
        meta.lee_noise_variance = self.lee_noise_variance

        # ---- Task 5: synthetic fallback ----
        real_available = (
            vv_path is not None
            and vh_path is not None
            and os.path.isfile(vv_path)
            and os.path.isfile(vh_path)
        )
        if not real_available:
            logger.info("Real SAR tile unavailable — generating synthetic SAR")
            return self._synthetic_fallback(meta)

        # ---- Task 1: load VV and VH GeoTIFFs ----
        vv, vv_crs, vv_res = self._load_sar_band(vv_path, "VV")
        vh, vh_crs, vh_res = self._load_sar_band(vh_path, "VH")

        meta.source_path_vv = vv_path
        meta.source_path_vh = vh_path
        meta.source_name = os.path.basename(vv_path)
        meta.source_crs = vv_crs
        meta.native_resolution_m = vv_res
        meta.original_shape = (vv.shape[0], vv.shape[1])

        # ---- Task 2: convert to dB ----
        vv_db, vh_db = self._to_db(vv, vh)
        meta.vv_db_min = float(np.min(vv_db))
        meta.vv_db_max = float(np.max(vv_db))
        meta.vh_db_min = float(np.min(vh_db))
        meta.vh_db_max = float(np.max(vh_db))

        logger.info(
            "VV dB range: [%.1f, %.1f]  |  VH dB range: [%.1f, %.1f]",
            meta.vv_db_min, meta.vv_db_max,
            meta.vh_db_min, meta.vh_db_max,
        )

        # ---- Task 3: Lee speckle filter ----
        vv_filtered = self._lee_filter(vv_db)
        vh_filtered = self._lee_filter(vh_db)

        # ---- Task 4: normalize and resample ----
        vv_norm = self._normalize_channel(vv_filtered)
        vh_norm = self._normalize_channel(vh_filtered)

        vv_resampled = self._resample_to_target(vv_norm)
        vh_resampled = self._resample_to_target(vh_norm)

        # Stack as (2, H, W): ch0=VV, ch1=VH
        sar_tensor = np.stack([vv_resampled, vh_resampled], axis=0).astype(np.float32)
        meta.output_shape = sar_tensor.shape

        return sar_tensor, meta

    # -----------------------------------------------------------------------
    # Task 1 — SAR GeoTIFF reader
    # -----------------------------------------------------------------------

    def _load_sar_band(
        self, path: str, pol: str
    ) -> Tuple[np.ndarray, str, Optional[float]]:
        """
        Load a single Sentinel-1 polarisation band.

        Returns (array (H, W), crs_string, resolution_m).
        """
        try:
            import rasterio
            from rasterio.warp import reproject, Resampling
            from rasterio.mask import mask as rio_mask
        except ImportError:
            raise ImportError(
                "rasterio is required for SAR GeoTIFF loading. "
                "Install with: conda install -c conda-forge rasterio"
            )

        with rasterio.open(path) as src:
            crs_str = str(src.crs)
            resolution_m = abs(src.transform.a) if src.transform else SAR_NATIVE_RESOLUTION_M
            band = src.read(1).astype(np.float32)  # (H, W)

            # Reproject to EPSG:4326 if needed
            if src.crs and src.crs.to_string() != TARGET_CRS:
                logger.info("Reprojecting %s from %s → %s", pol, src.crs, TARGET_CRS)
                band, _ = reproject(
                    source=band,
                    destination=np.zeros_like(band),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_crs=TARGET_CRS,
                    resampling=Resampling.bilinear,
                )

        logger.info("Loaded %s: shape=%s, resolution=%.1f m", pol, band.shape, resolution_m)
        return band, crs_str, resolution_m

    # -----------------------------------------------------------------------
    # Task 2 — dB conversion
    # -----------------------------------------------------------------------

    def _to_db(self, vv: np.ndarray, vh: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert linear power (sigma-nought) to dB scale.

        Formula:  dB = 10 * log10(linear_power + epsilon)

        Sentinel-1 GRD is typically delivered in linear power (sigma0).
        If the data is already in dB (values are negative), we skip.
        """
        # Detect if already dB: median < 0 strongly suggests dB scale
        vv_median = float(np.median(vv))
        if vv_median < 0:
            logger.info("VV median %.1f → data appears already in dB; skipping conversion", vv_median)
            return vv, vh

        logger.info("Converting linear power → dB (epsilon=%.1e)", DB_EPSILON)
        vv_db = 10.0 * np.log10(np.maximum(vv, DB_EPSILON))
        vh_db = 10.0 * np.log10(np.maximum(vh, DB_EPSILON))
        return vv_db, vh_db

    # -----------------------------------------------------------------------
    # Task 3 — Lee speckle filter
    # -----------------------------------------------------------------------

    def _lee_filter(self, image: np.ndarray) -> np.ndarray:
        """
        Lee sigma filter for speckle reduction (7×7 local window).

        Formula:
            filtered = mean_local + K * (pixel - mean_local)
            K = var_local / (var_local + var_noise)

        This is an edge-preserving adaptive filter: in homogeneous areas
        (var_local ≈ var_noise → K ≈ 0.5) it smooths; near edges
        (var_local >> var_noise → K ≈ 1) it preserves the pixel.

        scipy.ndimage.uniform_filter is used for the local mean/variance.
        """
        from scipy.ndimage import uniform_filter

        w = self.lee_window
        noise_var = self.lee_noise_variance

        # Local mean
        mean_local = uniform_filter(image, size=w)

        # Local variance: E[X²] - E[X]²
        mean_sq = uniform_filter(image ** 2, size=w)
        var_local = np.maximum(mean_sq - mean_local ** 2, 0.0)

        # Lee filter weight
        K = var_local / (var_local + noise_var)

        filtered = mean_local + K * (image - mean_local)

        logger.info(
            "Lee filter (%d×%d): K range [%.3f, %.3f], mean K=%.3f",
            w, w, float(K.min()), float(K.max()), float(K.mean()),
        )

        return filtered.astype(np.float32)

    def _median_filter(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Simpler alternative: median filter for speckle reduction.
        Included for comparison — Lee filter preserves road linearity better.
        """
        from scipy.ndimage import median_filter

        return median_filter(image, size=kernel_size).astype(np.float32)

    # -----------------------------------------------------------------------
    # Task 4 — Normalization and resampling
    # -----------------------------------------------------------------------

    def _normalize_channel(self, channel: np.ndarray) -> np.ndarray:
        """
        Percentile-based normalization to [0, 1].

        Uses 1st–99th percentile clipping to handle outlier speckle.
        """
        lo = float(np.percentile(channel, PERCENTILE_LOW))
        hi = float(np.percentile(channel, PERCENTILE_HIGH))
        rng = hi - lo
        if rng < 1e-8:
            rng = 1e-8

        normalized = np.clip((channel - lo) / rng, 0.0, 1.0)
        return normalized.astype(np.float32)

    def _resample_to_target(self, channel: np.ndarray) -> np.ndarray:
        """
        Resample from native SAR resolution to optical target resolution
        and snap to target_size (multiple of 32).

        Uses bilinear interpolation (cv2.INTER_LINEAR).
        """
        try:
            import cv2
        except ImportError:
            raise ImportError(
                "opencv-python is required for resampling. "
                "Install with: pip install opencv-python"
            )

        H, W = channel.shape
        scale = self.target_size / max(H, W)
        new_H = self._snap_to_multiple(int(round(H * scale)), 32)
        new_W = self._snap_to_multiple(int(round(W * scale)), 32)

        resampled = cv2.resize(channel, (new_W, new_H), interpolation=cv2.INTER_LINEAR)

        # Pad to exact target_size if needed
        pad_H = max(0, self.target_size - new_H)
        pad_W = max(0, self.target_size - new_W)
        if pad_H > 0 or pad_W > 0:
            pad_top = pad_H // 2
            pad_bottom = pad_H - pad_top
            pad_left = pad_W // 2
            pad_right = pad_W - pad_left
            resampled = np.pad(
                resampled,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode="reflect",
            )

        return resampled.astype(np.float32)

    # -----------------------------------------------------------------------
    # Task 5 — Synthetic SAR generator
    # -----------------------------------------------------------------------

    def _synthetic_fallback(self, meta: SARMeta) -> Tuple[np.ndarray, SARMeta]:
        """
        Generate a realistic synthetic SAR tile for Bengaluru.

        Physics-based approach:
        - Base texture: Rayleigh-distributed speckle (clutter background)
        - Roads: dark linear features (specular reflection → low backscatter)
        - Buildings: bright point-like clusters (double-bounce / corner reflectors)
        - Vegetation: medium-intensity textured patches

        NOTE: This is sufficient for testing the fusion pipeline (Phase 6)
        but NOT for training production models.
        """
        meta.is_synthetic = True
        meta.source_name = "synthetic_sentinel1_koramangala"
        meta.source_crs = TARGET_CRS
        meta.native_resolution_m = SAR_NATIVE_RESOLUTION_M
        meta.original_shape = (self.target_size, self.target_size)
        meta.vv_db_min = TYPICAL_URBAN_DB_RANGE[0]
        meta.vv_db_max = TYPICAL_URBAN_DB_RANGE[1]
        meta.vh_db_min = TYPICAL_URBAN_DB_RANGE[0]
        meta.vh_db_max = TYPICAL_URBAN_DB_RANGE[1]

        size = self.target_size
        rng = np.random.RandomState(42)

        # Step 1: Rayleigh speckle background (realistic SAR texture)
        # Rayleigh parameter controls brightness — urban ~0.3
        vv_speckle = rng.rayleigh(scale=0.3, size=(size, size)).astype(np.float32)
        vh_speckle = rng.rayleigh(scale=0.25, size=(size, size)).astype(np.float32)

        # Convert to dB for road/building painting
        vv_db = 10.0 * np.log10(np.maximum(vv_speckle, DB_EPSILON))
        vh_db = 10.0 * np.log10(np.maximum(vh_speckle, DB_EPSILON))

        # Step 2: Paint roads as dark linear features (~ -14 dB)
        # Main road: horizontal across middle
        road_width = 4
        vv_db[size // 2 - road_width : size // 2 + road_width, :] = -14.0
        vh_db[size // 2 - road_width : size // 2 + road_width, :] = -16.0

        # Cross road: vertical
        vv_db[:, size // 2 - road_width : size // 2 + road_width] = -13.5
        vh_db[:, size // 2 - road_width : size // 2 + road_width] = -15.5

        # Diagonal road (top-left to bottom-right)
        for i in range(-road_width, road_width + 1):
            diag_idx = np.arange(size)
            row_idx = np.clip(diag_idx + i + size // 4, 0, size - 1)
            col_idx = np.clip(diag_idx + i - size // 4, 0, size - 1)
            vv_db[row_idx, col_idx] = -14.5
            vh_db[row_idx, col_idx] = -16.5

        # Step 3: Paint buildings as bright clusters (double-bounce, ~ +2 dB)
        n_buildings = 15
        for _ in range(n_buildings):
            cx = rng.randint(size // 8, 7 * size // 8)
            cy = rng.randint(size // 8, 7 * size // 8)
            bw = rng.randint(3, 8)
            bh = rng.randint(3, 8)
            x0 = max(0, cx - bw // 2)
            x1 = min(size, cx + bw // 2)
            y0 = max(0, cy - bh // 2)
            y1 = min(size, cy + bh // 2)
            vv_db[y0:y1, x0:x1] = rng.uniform(0.5, 3.0)
            vh_db[y0:y1, x0:x1] = rng.uniform(-1.0, 1.5)

        # Step 4: Add vegetation patches (medium, textured)
        n_veg = 10
        for _ in range(n_veg):
            cx = rng.randint(0, size)
            cy = rng.randint(0, size)
            radius = rng.randint(8, 25)
            Y, X = np.ogrid[:size, :size]
            dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
            mask = dist < radius
            vv_db[mask] += rng.uniform(-1.0, 0.0)
            vh_db[mask] += rng.uniform(-0.5, 0.5)

        # Step 5: Lee filter pass to smooth speckle (like real preprocessed SAR)
        vv_filtered = self._lee_filter(vv_db)
        vh_filtered = self._lee_filter(vh_db)

        # Normalize
        vv_norm = self._normalize_channel(vv_filtered)
        vh_norm = self._normalize_channel(vh_filtered)

        sar_tensor = np.stack([vv_norm, vh_norm], axis=0).astype(np.float32)
        meta.output_shape = sar_tensor.shape

        logger.info(
            "Synthetic SAR: roads ~%.1f dB, buildings ~+%.1f dB, size=%d",
            TYPICAL_ROAD_DB_RANGE[0], 2.0, size,
        )
        return sar_tensor, meta

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _snap_to_multiple(value: int, base: int) -> int:
        """Snap value to the nearest multiple of *base*."""
        return max(base, int(base * round(value / base)))


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def preprocess_sar(
    vv_path: Optional[str] = None,
    vh_path: Optional[str] = None,
    target_size: int = 512,
    **kwargs,
) -> Tuple[np.ndarray, SARMeta]:
    """
    One-shot convenience wrapper.

    >>> tensor, meta = preprocess_sar()
    >>> print(tensor.shape)  # (2, 512, 512)
    """
    preprocessor = SARPreprocessor(target_size=target_size, **kwargs)
    return preprocessor.process(vv_path, vh_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 5 — SAR Preprocessing Pipeline")
    parser.add_argument("vv", nargs="?", default=None, help="Path to VV polarization GeoTIFF")
    parser.add_argument("vh", nargs="?", default=None, help="Path to VH polarization GeoTIFF")
    parser.add_argument("--target-size", type=int, default=512, help="Target spatial size (default: 512)")
    parser.add_argument("--lee-window", type=int, default=LEE_WINDOW_SIZE, help=f"Lee filter window (default: {LEE_WINDOW_SIZE})")
    parser.add_argument("--median", action="store_true", help="Use median filter instead of Lee")
    args = parser.parse_args()

    preprocessor = SARPreprocessor(target_size=args.target_size, lee_window=args.lee_window)
    tensor, meta = preprocessor.process(args.vv, args.vh)

    print(f"Tensor shape:     {tensor.shape}")
    print(f"Source:           {meta.source_name}")
    print(f"Synthetic:        {meta.is_synthetic}")
    print(f"VV dB range:      [{meta.vv_db_min:.1f}, {meta.vv_db_max:.1f}]")
    print(f"VH dB range:      [{meta.vh_db_min:.1f}, {meta.vh_db_max:.1f}]")
    print(f"Native res:       {meta.native_resolution_m} m")
    print(f"Target res:       {meta.target_resolution_m} m")
    print(f"Lee window:       {meta.lee_window}×{meta.lee_window}")
    print(f"CRS:              {meta.source_crs}")
    print(f"Warnings:         {meta.warnings}")
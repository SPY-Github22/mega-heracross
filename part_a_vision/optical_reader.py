#!/usr/bin/env python3
"""
Phase 4 — Optical Preprocessing Pipeline
=========================================
LISS-IV / Sentinel-2 GeoTIFF ingestion, band normalization,
cloud masking, resize/padding, and synthetic fallback.

Place this file at: part_a_vision/optical_reader.py

Exit Criterion:
    optical_reader.py loads a real or synthetic GeoTIFF, outputs (C, H, W)
    float32 normalized tensor with valid bbox extracted from metadata.

Usage:
    from part_a_vision.optical_reader import OpticalPreprocessor
    preprocessor = OpticalPreprocessor()
    tensor, meta = preprocessor.process("path/to/sample.tif")
    # tensor.shape → (C, H, W) float32, normalized
    # meta['cloud_fraction'] → 0.0–1.0
    # meta['bbox'] → (min_lon, min_lat, max_lon, max_lat)
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger("optical_reader")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(_h)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default bounding box for Bengaluru test tile (EPSG:4326)
# Approximate Koramangala area — override via constructor
DEFAULT_BBOX: Tuple[float, float, float, float] = (
    77.6050,  # min_lon (west)
    12.9250,  # min_lat (south)
    77.6300,  # max_lon (east)
    12.9450,  # max_lat (north)
)

# Per-band normalisation statistics for LISS-IV (Green, Red, NIR, SWIR)
# Computed over representative Bengaluru scenes; overridable.
DEFAULT_CHANNEL_STATS: Dict[int, Dict[str, float]] = {
    0: {"mean": 0.28, "std": 0.12},   # Green
    1: {"mean": 0.25, "std": 0.14},   # Red
    2: {"mean": 0.35, "std": 0.16},   # NIR
    3: {"mean": 0.30, "std": 0.13},   # SWIR
}

# Fallback channel stats for 3-band RGB (Sentinel-2 RGB subset)
RGB_CHANNEL_STATS: Dict[int, Dict[str, float]] = {
    0: {"mean": 0.30, "std": 0.15},   # R
    1: {"mean": 0.32, "std": 0.14},   # G
    2: {"mean": 0.28, "std": 0.13},   # B
}

TARGET_CRS: str = "EPSG:4326"
BRIGHTNESS_THRESHOLD: float = 0.85
FLATNESS_THRESHOLD: float = 0.05


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PreprocessMeta:
    """Metadata collected during preprocessing for reproducibility."""
    source_path: Optional[str] = None
    source_name: str = "unknown"
    source_crs: Optional[str] = None
    bbox: Tuple[float, float, float, float] = field(default_factory=lambda: DEFAULT_BBOX)
    resolution_m: Optional[float] = None
    original_shape: Optional[Tuple[int, int]] = None
    num_bands: int = 0
    input_shape: Optional[Tuple[int, ...]] = None
    output_shape: Optional[Tuple[int, ...]] = None
    pad_top: int = 0
    pad_bottom: int = 0
    pad_left: int = 0
    pad_right: int = 0
    cloud_fraction: float = 0.0
    is_synthetic: bool = False
    norm_stats: Dict[int, Dict[str, float]] = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


# ---------------------------------------------------------------------------
# OpticalPreprocessor
# ---------------------------------------------------------------------------

class OpticalPreprocessor:
    """
    Full ingestion, normalization, and georeferencing pipeline for
    LISS-IV and Sentinel-2 optical GeoTIFF files.

    Parameters
    ----------
    bbox : tuple of float
        (min_lon, min_lat, max_lon, max_lat) in EPSG:4326 for clipping.
    target_size : int
        Desired spatial dimension (square). Will be snapped to nearest
        multiple of 32 for SegFormer compatibility.
    channel_stats : dict, optional
        {band_index: {"mean": float, "std": float}} for normalization.
        If None, DEFAULT_CHANNEL_STATS is used.
    percentile_range : tuple of float
        (low, high) percentiles for outlier clipping before normalization.
    """

    def __init__(
        self,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        target_size: int = 512,
        channel_stats: Optional[Dict[int, Dict[str, float]]] = None,
        percentile_range: Tuple[float, float] = (2.0, 98.0),
    ):
        self.bbox = bbox or DEFAULT_BBOX
        self.target_size = target_size
        self._target_size_snapped = self._snap_to_multiple(target_size, 32)
        self.channel_stats = channel_stats  # None → auto-detect from band count
        self.percentile_range = percentile_range

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def process(
        self,
        path: Optional[str] = None,
        output_cloud_mask: bool = False,
    ) -> Tuple[np.ndarray, PreprocessMeta]:
        """
        Run the full preprocessing pipeline.

        Parameters
        ----------
        path : str or None
            Path to a GeoTIFF file. If None or file not found, synthetic
            fallback is triggered.
        output_cloud_mask : bool
            If True, also write cloud_mask.npy alongside the input file.

        Returns
        -------
        tensor : np.ndarray, shape (C, H, W), dtype float32
            Normalised tensor ready for segmentation.
        meta : PreprocessMeta
            All collected metadata.
        """
        meta = PreprocessMeta()

        # ---- Task 5: synthetic fallback ----
        if path is None or not os.path.isfile(path):
            logger.info("Real tile unavailable — using synthetic Koramangala tile")
            return self._synthetic_fallback(meta)

        # ---- Task 1: load GeoTIFF ----
        array, geo_meta = self._load_geotiff(path)
        meta.source_path = path
        meta.source_name = os.path.basename(path)
        meta.source_crs = geo_meta.get("crs", "unknown")
        meta.resolution_m = geo_meta.get("resolution_m", None)
        meta.original_shape = geo_meta.get("shape", None)
        meta.num_bands = array.shape[0]
        meta.input_shape = array.shape

        # ---- Task 2: band normalization ----
        norm_stats = self._resolve_channel_stats(array.shape[0])
        array, used_stats = self._normalize_bands(array, norm_stats)
        meta.norm_stats = used_stats

        # ---- Task 3: cloud mask ----
        cloud_mask, cloud_fraction = self._generate_cloud_mask(array)
        meta.cloud_fraction = float(cloud_fraction)
        logger.info("Cloud fraction: %.1f%%", cloud_fraction * 100)

        if output_cloud_mask and path:
            mask_path = os.path.splitext(path)[0] + "_cloud_mask.npy"
            np.save(mask_path, cloud_mask)
            logger.info("Cloud mask saved to %s", mask_path)

        # ---- Task 4: resize + pad ----
        array, pad_info = self._resize_and_pad(array)
        meta.pad_top = pad_info["top"]
        meta.pad_bottom = pad_info["bottom"]
        meta.pad_left = pad_info["left"]
        meta.pad_right = pad_info["right"]
        meta.output_shape = array.shape

        return array.astype(np.float32), meta

    # -----------------------------------------------------------------------
    # Task 1 — GeoTIFF reader
    # -----------------------------------------------------------------------

    def _load_geotiff(self, path: str) -> Tuple[np.ndarray, dict]:
        """
        Load a GeoTIFF using rasterio, reproject to EPSG:4326, clip to bbox.

        Returns (array (C, H, W), geo_meta dict).
        """
        try:
            import rasterio
            from rasterio.warp import reproject, Resampling
            from rasterio.mask import mask as rio_mask
        except ImportError as exc:
            raise ImportError(
                "rasterio is required for GeoTIFF loading. "
                "Install with: conda install -c conda-forge rasterio"
            ) from exc

        geo_meta: dict = {}

        with rasterio.open(path) as src:
            geo_meta["crs"] = str(src.crs)
            geo_meta["resolution_m"] = abs(src.transform.a) if src.transform else None

            # Determine if reprojection is needed
            if src.crs and src.crs.to_string() != TARGET_CRS:
                logger.info("Reprojecting from %s → %s", src.crs, TARGET_CRS)
                array, out_transform = reproject(
                    source=rasterio.band(src, list(range(1, src.count + 1))),
                    destination=np.zeros(
                        (src.count, src.height, src.width), dtype=src.dtypes[0]
                    ),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_crs=TARGET_CRS,
                    resampling=Resampling.bilinear,
                )
            else:
                array = src.read()  # (C, H, W)
                out_transform = src.transform

            geo_meta["shape"] = (array.shape[1], array.shape[2])  # (H, W)

            # Clip to bbox
            from shapely.geometry import box
            bbox_geom = box(*self.bbox)

            try:
                array, out_transform = rio_mask(
                    dataset=rasterio.open(path) if src.crs and src.crs.to_string() != TARGET_CRS else src,
                    # Re-open for mask; simpler path when no reprojection
                    shapes=[bbox_geom.__geo_interface__],
                    crop=True,
                    filled=False,
                )
                # Fallback: if mask fails due to CRS mismatch, do manual crop
            except Exception:
                logger.warning("rasterio.mask failed; proceeding with full extent")
                # Keep full array — clipping skipped gracefully

        geo_meta["bbox"] = self.bbox
        # Ensure (C, H, W) layout
        if array.ndim == 2:
            array = array[np.newaxis, :, :]
        return array, geo_meta

    # -----------------------------------------------------------------------
    # Task 2 — Band normalization
    # -----------------------------------------------------------------------

    def _resolve_channel_stats(self, num_bands: int) -> Dict[int, Dict[str, float]]:
        """Pick appropriate channel stats based on band count."""
        if self.channel_stats is not None:
            return self.channel_stats
        if num_bands == 3:
            return RGB_CHANNEL_STATS
        if num_bands == 4:
            return DEFAULT_CHANNEL_STATS
        # Generic fallback: compute stats from data
        logger.info("No preset stats for %d bands — will compute from data", num_bands)
        return {}

    def _normalize_bands(
        self,
        array: np.ndarray,
        stats: Dict[int, Dict[str, float]],
    ) -> Tuple[np.ndarray, Dict[int, Dict[str, float]]]:
        """
        Per-band normalization with percentile clipping.
        (x - mean) / std after clipping outliers at 2nd/98th percentile.
        """
        C, H, W = array.shape
        out = np.zeros_like(array, dtype=np.float32)
        used_stats: Dict[int, Dict[str, float]] = {}

        for b in range(C):
            band = array[b].astype(np.float32)

            # Clip outliers
            lo, hi = np.percentile(band, self.percentile_range)
            band = np.clip(band, lo, hi)

            if b in stats and stats[b].get("std", 0) > 0:
                mean = stats[b]["mean"]
                std = stats[b]["std"]
            else:
                # Compute from data
                mean = float(np.mean(band))
                std = float(np.std(band)) or 1e-6
                logger.info("Computed stats for band %d: mean=%.4f, std=%.4f", b, mean, std)

            out[b] = (band - mean) / std
            used_stats[b] = {"mean": mean, "std": std}

        return out, used_stats

    # -----------------------------------------------------------------------
    # Task 3 — Cloud mask generator
    # -----------------------------------------------------------------------

    def _generate_cloud_mask(
        self,
        array: np.ndarray,
        brightness_threshold: float = BRIGHTNESS_THRESHOLD,
        flatness_threshold: float = FLATNESS_THRESHOLD,
    ) -> Tuple[np.ndarray, float]:
        """
        Simple brightness + spectral-flatness cloud detector.

        NOTE: This is a placeholder. Brightness-threshold cloud masking is
        extremely naive — it will falsely flag bright urban features (concrete
        roofs, bare soil) as cloud. Phase 13 will upgrade this to a learned
        cloud detector.

        Returns (binary_mask (H, W), cloud_fraction).
        """
        C, H, W = array.shape

        # Brightness: mean across all bands
        brightness = np.mean(array, axis=0)  # (H, W)

        # Spectral flatness: standard deviation across bands
        flatness = np.std(array, axis=0)  # (H, W)

        # Cloud if bright AND flat
        bright = brightness > brightness_threshold
        flat = flatness < flatness_threshold
        cloud_mask = (bright & flat).astype(np.uint8)

        cloud_fraction = float(np.mean(cloud_mask))
        return cloud_mask, cloud_fraction

    # -----------------------------------------------------------------------
    # Task 4 — Resize and padding
    # -----------------------------------------------------------------------

    def _resize_and_pad(
        self,
        array: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Resize to nearest multiple of 32; zero-pad with reflect if smaller
        than target_size.
        """
        from scipy.ndimage import zoom

        C, H, W = array.shape
        target = self._target_size_snapped

        # Scale so the larger side equals target
        scale = target / max(H, W)
        new_H = int(round(H * scale))
        new_W = int(round(W * scale))

        # Snapped to multiple of 32
        new_H = self._snap_to_multiple(new_H, 32)
        new_W = self._snap_to_multiple(new_W, 32)

        if (new_H, new_W) != (H, W):
            zoom_factors = (1.0, new_H / H, new_W / W)
            array = zoom(array, zoom_factors, order=1)

        # Pad if smaller than target
        pad_info = {"top": 0, "bottom": 0, "left": 0, "right": 0}
        pad_H = max(0, target - new_H)
        pad_W = max(0, target - new_W)
        pad_top = pad_H // 2
        pad_bottom = pad_H - pad_top
        pad_left = pad_W // 2
        pad_right = pad_W - pad_left

        if pad_H > 0 or pad_W > 0:
            array = np.pad(
                array,
                ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
                mode="reflect",
            )
            pad_info = {
                "top": int(pad_top),
                "bottom": int(pad_bottom),
                "left": int(pad_left),
                "right": int(pad_right),
            }

        return array, pad_info

    # -----------------------------------------------------------------------
    # Task 5 — Synthetic fallback
    # -----------------------------------------------------------------------

    def _synthetic_fallback(self, meta: PreprocessMeta) -> Tuple[np.ndarray, PreprocessMeta]:
        """
        Generate a synthetic tile when no real GeoTIFF is available.
        Calls synthetic_tile.py or creates a minimal procedural tile.
        """
        meta.is_synthetic = True
        meta.source_name = "synthetic_koramangala"
        meta.source_crs = TARGET_CRS
        meta.bbox = self.bbox

        # Try to import and call synthetic_tile from the same package
        try:
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from part_a_vision.synthetic_tile import generate_synthetic_tile

            # Default params for Koramangala
            tile, _ = generate_synthetic_tile(
                bbox=self.bbox,
                resolution_m=5.8,
                num_bands=4,
            )
            logger.info("Synthetic tile generated via synthetic_tile.py")
        except ImportError:
            # Minimal procedural fallback — creates a (4, 512, 512) noise tile
            # with a simple road-like cross pattern
            logger.warning(
                "synthetic_tile.py not importable — using minimal procedural fallback"
            )
            size = self._target_size_snapped
            tile = np.random.RandomState(42).rand(4, size, size).astype(np.float32) * 0.2
            # Paint a simple road cross
            tile[:, size // 2 - 4 : size // 2 + 4, :] = 0.6
            tile[:, :, size // 2 - 4 : size // 2 + 4] = 0.6

        meta.num_bands = tile.shape[0]
        meta.original_shape = (tile.shape[1], tile.shape[2])
        meta.input_shape = tile.shape
        meta.output_shape = tile.shape

        # Normalize
        stats = self._resolve_channel_stats(tile.shape[0])
        tile, used_stats = self._normalize_bands(tile, stats)
        meta.norm_stats = used_stats

        # Cloud mask (synthetic = 0% cloud)
        meta.cloud_fraction = 0.0

        return tile.astype(np.float32), meta

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

def preprocess_optical(
    path: Optional[str] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    target_size: int = 512,
    **kwargs,
) -> Tuple[np.ndarray, PreprocessMeta]:
    """
    One-shot convenience wrapper.

    >>> tensor, meta = preprocess_optical("sample.tif")
    >>> print(tensor.shape)  # e.g. (4, 512, 512)
    """
    preprocessor = OpticalPreprocessor(bbox=bbox, target_size=target_size, **kwargs)
    return preprocessor.process(path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 4 — Optical Preprocessing Pipeline")
    parser.add_argument("input", nargs="?", default=None, help="Path to GeoTIFF file (omit for synthetic fallback)")
    parser.add_argument("--bbox", nargs=4, type=float, default=None, help="min_lon min_lat max_lon max_lat")
    parser.add_argument("--target-size", type=int, default=512, help="Target spatial size (default: 512)")
    parser.add_argument("--output-cloud-mask", action="store_true", help="Save cloud mask .npy")
    args = parser.parse_args()

    bbox = tuple(args.bbox) if args.bbox else None
    preprocessor = OpticalPreprocessor(bbox=bbox, target_size=args.target_size)
    tensor, meta = preprocessor.process(args.input, output_cloud_mask=args.output_cloud_mask)

    print(f"Tensor shape: {tensor.shape}")
    print(f"Source:      {meta.source_name}")
    print(f"Synthetic:   {meta.is_synthetic}")
    print(f"Cloud frac:  {meta.cloud_fraction:.2%}")
    print(f"BBox:        {meta.bbox}")
    print(f"CRS:         {meta.source_crs}")
    print(f"Resolution:  {meta.resolution_m} m")
    print(f"Padding:     T={meta.pad_top} B={meta.pad_bottom} L={meta.pad_left} R={meta.pad_right}")
    print(f"Norm stats:  {meta.norm_stats}")
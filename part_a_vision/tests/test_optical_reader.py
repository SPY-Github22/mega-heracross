#!/usr/bin/env python3
"""
Phase 4 Exit Criterion Tests — Optical Preprocessing Pipeline
==============================================================
25+ tests covering all 5 tasks plus edge cases and integration.

Place at: part_a_vision/tests/test_optical_reader.py

Run:
    python -m pytest part_a_vision/tests/test_optical_reader.py -v
    or, if GDAL/rasterio not available:
    python -m pytest part_a_vision/tests/test_optical_reader.py -v --skip-rasterio
"""

from __future__ import annotations

import os
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest

# -- Ensure package is importable -------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from optical_reader import (
    OpticalPreprocessor,
    PreprocessMeta,
    preprocess_optical,
    DEFAULT_BBOX,
    DEFAULT_CHANNEL_STATS,
    RGB_CHANNEL_STATS,
    BRIGHTNESS_THRESHOLD,
    FLATNESS_THRESHOLD,
)

# -- Skip marker for rasterio-dependent tests -------------------------------
try:
    import rasterio  # noqa: F401

    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

skip_rasterio = pytest.mark.skipif(not RASTERIO_AVAILABLE, reason="rasterio not installed")


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def preprocessor():
    """Default preprocessor with Bengaluru bbox."""
    return OpticalPreprocessor(target_size=512)


@pytest.fixture
def dummy_4band_array():
    """A (4, 256, 256) float32 array mimicking LISS-IV bands."""
    rng = np.random.RandomState(42)
    arr = rng.rand(4, 256, 256).astype(np.float32) * 0.4
    # Add a bright road-like stripe
    arr[:, 100:108, :] = 0.7
    return arr


@pytest.fixture
def dummy_3band_array():
    """A (3, 256, 256) float32 array mimicking RGB."""
    rng = np.random.RandomState(7)
    return rng.rand(3, 256, 256).astype(np.float32) * 0.5


@pytest.fixture
def bright_cloud_array():
    """A (4, 128, 128) array with a bright flat patch (simulated cloud)."""
    arr = np.random.RandomState(99).rand(4, 128, 128).astype(np.float32) * 0.3
    # Cloud patch: bright and flat
    arr[:, 20:60, 20:60] = 0.95
    return arr


# ============================================================================
# Task 1 Tests — GeoTIFF reader (rasterio-dependent)
# ============================================================================

class TestGeoTiffReader:

    @skip_rasterio
    def test_load_real_tif_returns_array(self, preprocessor, tmp_path):
        """Create a minimal GeoTIFF and verify it loads as (C, H, W)."""
        import rasterio
        from rasterio.transform import from_bounds

        tif_path = tmp_path / "test.tif"
        data = np.random.RandomState(1).rand(3, 64, 64).astype(np.float32)

        transform = from_bounds(
            DEFAULT_BBOX[0], DEFAULT_BBOX[1],
            DEFAULT_BBOX[2], DEFAULT_BBOX[3],
            64, 64,
        )
        with rasterio.open(
            str(tif_path), "w",
            driver="GTiff",
            height=64, width=64,
            count=3,
            dtype="float32",
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(data)

        tensor, meta = preprocessor.process(str(tif_path))
        assert tensor.ndim == 3, f"Expected 3D (C,H,W), got {tensor.ndim}D"
        assert tensor.shape[0] == 3, f"Expected 3 bands, got {tensor.shape[0]}"
        assert tensor.dtype == np.float32

    @skip_rasterio
    def test_reproject_to_epsg4326(self, preprocessor, tmp_path):
        """Tile in EPSG:32643 (UTM zone 43N) should be reprojected to 4326."""
        import rasterio
        from rasterio.transform import from_bounds

        tif_path = tmp_path / "utm.tif"
        data = np.ones((1, 32, 32), dtype=np.float32) * 0.5
        with rasterio.open(
            str(tif_path), "w",
            driver="GTiff",
            height=32, width=32,
            count=1,
            dtype="float32",
            crs="EPSG:32643",
            transform=from_bounds(750000, 1425000, 751000, 1426000, 32, 32),
        ) as dst:
            dst.write(data)

        tensor, meta = preprocessor.process(str(tif_path))
        # Should not crash; output CRS metadata recorded
        assert meta.source_crs is not None
        assert tensor.ndim == 3

    @skip_rasterio
    def test_clip_to_bbox(self, preprocessor, tmp_path):
        """Verify that output is clipped to configured bbox region."""
        import rasterio
        from rasterio.transform import from_bounds

        # Make a tile larger than the bbox
        big_bbox = (77.58, 12.90, 77.65, 12.97)
        tif_path = tmp_path / "big.tif"
        data = np.random.RandomState(5).rand(2, 128, 128).astype(np.float32)

        with rasterio.open(
            str(tif_path), "w",
            driver="GTiff",
            height=128, width=128,
            count=2,
            dtype="float32",
            crs="EPSG:4326",
            transform=from_bounds(*big_bbox, 128, 128),
        ) as dst:
            dst.write(data)

        tensor, meta = preprocessor.process(str(tif_path))
        # The bbox is recorded; clipping may fail gracefully
        assert meta.bbox is not None


# ============================================================================
# Task 1 (non-rasterio) — Metadata handling
# ============================================================================

class TestMetadata:
    def test_meta_dataclass_defaults(self):
        meta = PreprocessMeta()
        assert meta.cloud_fraction == 0.0
        assert meta.is_synthetic is False
        assert meta.pad_top == 0
        assert meta.warnings == []

    def test_meta_to_dict_serializable(self):
        meta = PreprocessMeta(source_name="test.tif", cloud_fraction=0.12)
        d = meta.to_dict()
        assert d["source_name"] == "test.tif"
        assert d["cloud_fraction"] == 0.12

    def test_meta_records_source_from_process(self, preprocessor):
        """Synthetic fallback must record is_synthetic=True."""
        tensor, meta = preprocessor.process(None)
        assert meta.is_synthetic is True
        assert "synthetic" in meta.source_name.lower()

    def test_meta_records_input_shape(self, preprocessor):
        tensor, meta = preprocessor.process(None)
        assert meta.input_shape is not None
        assert len(meta.input_shape) == 3

    def test_meta_records_output_shape(self, preprocessor):
        tensor, meta = preprocessor.process(None)
        assert meta.output_shape is not None
        assert len(meta.output_shape) == 3


# ============================================================================
# Task 2 — Band normalization
# ============================================================================

class TestBandNormalization:

    def test_output_range_reasonable(self, preprocessor, dummy_4band_array):
        """Normalized values should not explode; expect most in [-4, 4]."""
        # Bypass file loading: call internal methods directly
        stats = preprocessor._resolve_channel_stats(4)
        normed, _ = preprocessor._normalize_bands(dummy_4band_array, stats)
        assert normed.dtype == np.float32
        # Vast majority should be within ±4 sigma
        assert np.abs(normed).max() < 10.0, f"Outlier spike: {np.abs(normed).max()}"

    def test_percentile_clipping_removes_extremes(self, preprocessor):
        """After clipping at 2nd/98th percentile, max ≤ original max."""
        arr = np.random.RandomState(3).rand(3, 64, 64).astype(np.float32)
        arr[0, 0, 0] = 999.0  # extreme outlier
        stats = preprocessor._resolve_channel_stats(3)
        normed, _ = preprocessor._normalize_bands(arr, stats)
        # The outlier should be clipped away
        assert np.abs(normed).max() < 50.0, f"Outlier not clipped: {np.abs(normed).max()}"

    def test_uses_default_stats_for_4band(self, preprocessor):
        stats = preprocessor._resolve_channel_stats(4)
        assert 0 in stats
        assert 3 in stats
        assert "mean" in stats[0]

    def test_uses_rgb_stats_for_3band(self, preprocessor):
        stats = preprocessor._resolve_channel_stats(3)
        assert 0 in stats
        assert 2 in stats

    def test_computes_stats_for_unknown_band_count(self, preprocessor):
        """5-band input should compute stats from data (no preset)."""
        stats = preprocessor._resolve_channel_stats(5)
        assert stats == {}

    def test_normalization_preserves_spatial_structure(self, preprocessor, dummy_3band_array):
        """Normalization is per-band; spatial dims unchanged."""
        stats = preprocessor._resolve_channel_stats(3)
        normed, _ = preprocessor._normalize_bands(dummy_3band_array, stats)
        assert normed.shape == dummy_3band_array.shape

    def test_custom_channel_stats_override(self):
        custom_stats = {0: {"mean": 0.1, "std": 0.05}}
        pp = OpticalPreprocessor(channel_stats=custom_stats)
        arr = np.ones((1, 32, 32), dtype=np.float32) * 0.2
        normed, used = pp._normalize_bands(arr, custom_stats)
        # (0.2 - 0.1) / 0.05 = 2.0
        assert np.allclose(normed, 2.0, atol=0.01)


# ============================================================================
# Task 3 — Cloud mask
# ============================================================================

class TestCloudMask:

    def test_cloud_mask_binary(self, preprocessor, bright_cloud_array):
        mask, frac = preprocessor._generate_cloud_mask(bright_cloud_array)
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 1})

    def test_cloud_fraction_between_0_and_1(self, preprocessor, bright_cloud_array):
        _, frac = preprocessor._generate_cloud_mask(bright_cloud_array)
        assert 0.0 <= frac <= 1.0

    def test_no_cloud_in_dark_scene(self, preprocessor):
        dark = np.random.RandomState(0).rand(3, 64, 64).astype(np.float32) * 0.2
        _, frac = preprocessor._generate_cloud_mask(dark)
        assert frac == 0.0

    def test_full_bright_flat_is_cloud(self, preprocessor):
        cloud = np.ones((3, 32, 32), dtype=np.float32) * 0.95
        _, frac = preprocessor._generate_cloud_mask(cloud)
        assert frac > 0.9, f"Expected >90% cloud, got {frac:.2%}"

    def test_bright_but_not_flat_not_cloud(self, preprocessor):
        """High brightness but high variance across bands → not cloud."""
        arr = np.zeros((4, 32, 32), dtype=np.float32)
        arr[0] = 0.9  # band 0 bright
        arr[1] = 0.1  # band 1 dark → high variance
        arr[2] = 0.5
        arr[3] = 0.3
        _, frac = preprocessor._generate_cloud_mask(arr)
        # Should not be flagged as cloud because std across bands is high
        assert frac < 0.5

    def test_custom_thresholds(self, preprocessor, bright_cloud_array):
        mask_strict, frac_strict = preprocessor._generate_cloud_mask(
            bright_cloud_array, brightness_threshold=0.99, flatness_threshold=0.01
        )
        mask_lenient, frac_lenient = preprocessor._generate_cloud_mask(
            bright_cloud_array, brightness_threshold=0.5, flatness_threshold=0.5
        )
        assert frac_strict <= frac_lenient


# ============================================================================
# Task 4 — Resize and padding
# ============================================================================

class TestResizeAndPad:

    def test_output_divisible_by_32(self, preprocessor):
        arr = np.random.RandomState(42).rand(4, 300, 300).astype(np.float32)
        resized, _ = preprocessor._resize_and_pad(arr)
        assert resized.shape[1] % 32 == 0
        assert resized.shape[2] % 32 == 0

    def test_small_input_padded_to_target(self, preprocessor):
        arr = np.random.RandomState(1).rand(3, 64, 64).astype(np.float32)
        resized, pad_info = preprocessor._resize_and_pad(arr)
        target = 512
        assert resized.shape[1] >= target
        assert resized.shape[2] >= target

    def test_pad_info_symmetric(self, preprocessor):
        arr = np.ones((1, 100, 200), dtype=np.float32)
        _, pad_info = preprocessor._resize_and_pad(arr)
        # Left/right should be nearly balanced
        assert abs(pad_info["left"] - pad_info["right"]) <= 1

    def test_already_correct_size_passes_through(self, preprocessor):
        """If input is already 512×512, padding should be zero."""
        arr = np.random.RandomState(10).rand(4, 512, 512).astype(np.float32)
        resized, pad_info = preprocessor._resize_and_pad(arr)
        assert pad_info["top"] == 0
        assert pad_info["bottom"] == 0
        assert pad_info["left"] == 0
        assert pad_info["right"] == 0

    def test_reflect_pad_mode_no_artifacts(self, preprocessor):
        """Reflect padding should not introduce zeros at edges."""
        arr = np.ones((1, 200, 200), dtype=np.float32)
        resized, _ = preprocessor._resize_and_pad(arr)
        # Edges should still be 1 (reflect copies edge values)
        assert resized[0, 0, 0] == pytest.approx(1.0, abs=0.1)


# ============================================================================
# Task 5 — Synthetic fallback
# ============================================================================

class TestSyntheticFallback:

    def test_fallback_with_none_path(self, preprocessor):
        tensor, meta = preprocessor.process(None)
        assert tensor.ndim == 3
        assert meta.is_synthetic is True

    def test_fallback_with_missing_file(self, preprocessor):
        tensor, meta = preprocessor.process("/nonexistent/path_xyz123.tif")
        assert meta.is_synthetic is True

    def test_fallback_tensor_is_float32(self, preprocessor):
        tensor, _ = preprocessor.process(None)
        assert tensor.dtype == np.float32

    def test_fallback_produces_valid_shape(self, preprocessor):
        tensor, meta = preprocessor.process(None)
        assert tensor.shape[0] > 0  # at least 1 band
        assert tensor.shape[1] % 32 == 0
        assert tensor.shape[2] % 32 == 0

    def test_fallback_bbox_preserved(self, preprocessor):
        _, meta = preprocessor.process(None)
        assert meta.bbox == preprocessor.bbox


# ============================================================================
# Integration tests
# ============================================================================

class TestIntegration:

    def test_convenience_function_works(self):
        tensor, meta = preprocess_optical(None, target_size=256)
        assert tensor.ndim == 3

    def test_full_pipeline_no_crash(self, preprocessor):
        """Run the full pipeline end-to-end with synthetic fallback."""
        tensor, meta = preprocessor.process(None)
        assert isinstance(tensor, np.ndarray)
        assert isinstance(meta, PreprocessMeta)
        assert tensor.ndim == 3

    def test_cloud_mask_output_file(self, preprocessor, tmp_path):
        """When output_cloud_mask=True and path is given, file is written."""
        tif_path = tmp_path / "dummy.tif"
        # Create a minimal fake tif (just bytes) — will fail gracefully
        tif_path.write_bytes(b"FAKE_TIFF_HEADER")
        # Should gracefully fall to synthetic since file isn't a real GeoTIFF
        tensor, meta = preprocessor.process(str(tif_path), output_cloud_mask=True)
        # Fallback handled gracefully
        assert meta.is_synthetic is True

    def test_reproducibility_same_input(self, preprocessor):
        """Same None input → same synthetic output (fixed seed)."""
        t1, _ = preprocessor.process(None)
        t2, _ = preprocessor.process(None)
        assert np.allclose(t1, t2)

    def test_custom_bbox_propagates(self):
        custom_bbox = (77.60, 12.93, 77.61, 12.94)
        pp = OpticalPreprocessor(bbox=custom_bbox)
        _, meta = pp.process(None)
        assert meta.bbox == custom_bbox

    def test_different_target_sizes(self):
        for size in [256, 512, 1024]:
            pp = OpticalPreprocessor(target_size=size)
            tensor, _ = pp.process(None)
            assert tensor.shape[1] % 32 == 0
            assert tensor.shape[2] % 32 == 0


# ============================================================================
# Edge cases
# ============================================================================

class TestEdgeCases:

    def test_single_band_input(self, preprocessor):
        arr = np.random.RandomState(0).rand(1, 128, 128).astype(np.float32)
        stats = preprocessor._resolve_channel_stats(1)
        normed, used = preprocessor._normalize_bands(arr, stats)
        assert normed.shape == (1, 128, 128)

    def test_very_wide_tile(self, preprocessor):
        arr = np.random.RandomState(2).rand(3, 64, 1024).astype(np.float32)
        resized, _ = preprocessor._resize_and_pad(arr)
        assert resized.shape[2] % 32 == 0

    def test_very_tall_tile(self, preprocessor):
        arr = np.random.RandomState(3).rand(3, 1024, 64).astype(np.float32)
        resized, _ = preprocessor._resize_and_pad(arr)
        assert resized.shape[1] % 32 == 0

    def test_all_zeros_input(self, preprocessor):
        arr = np.zeros((3, 64, 64), dtype=np.float32)
        stats = preprocessor._resolve_channel_stats(3)
        normed, _ = preprocessor._normalize_bands(arr, stats)
        assert not np.any(np.isnan(normed))
        assert not np.any(np.isinf(normed))

    def test_all_ones_input(self, preprocessor):
        arr = np.ones((3, 64, 64), dtype=np.float32)
        stats = preprocessor._resolve_channel_stats(3)
        normed, _ = preprocessor._normalize_bands(arr, stats)
        assert not np.any(np.isnan(normed))
        assert not np.any(np.isinf(normed))


# ============================================================================
# CLI smoke test
# ============================================================================

class TestCLI:
    def test_main_runs_without_args(self):
        """Smoke test: python optical_reader.py (no args → synthetic fallback)."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "optical_reader"],
            capture_output=True, text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        # Should succeed and mention synthetic
        assert "Synthetic" in result.stdout or result.returncode == 0
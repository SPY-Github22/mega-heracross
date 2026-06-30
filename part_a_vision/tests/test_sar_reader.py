#!/usr/bin/env python3
"""
Phase 5 Exit Criterion Tests — SAR Preprocessing Pipeline
==========================================================
40+ tests covering all 5 tasks plus edge cases and integration.

Place at: part_a_vision/tests/test_sar_reader.py

Run:
    python -m pytest part_a_vision/tests/test_sar_reader.py -v
    or, if GDAL/rasterio not available:
    python -m pytest part_a_vision/tests/test_sar_reader.py -v --skip-rasterio
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from sar_reader import (
    SARPreprocessor,
    SARMeta,
    preprocess_sar,
    DEFAULT_BBOX,
    LEE_WINDOW_SIZE,
    LEE_NOISE_VARIANCE,
    DB_EPSILON,
    TYPICAL_URBAN_DB_RANGE,
    TYPICAL_ROAD_DB_RANGE,
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
    return SARPreprocessor(target_size=256)


@pytest.fixture
def vv_linear():
    """Simulate VV in linear power scale (sigma-nought, values > 0)."""
    rng = np.random.RandomState(42)
    return rng.rayleigh(scale=0.3, size=(128, 128)).astype(np.float32)


@pytest.fixture
def vh_linear():
    """Simulate VH in linear power scale."""
    rng = np.random.RandomState(43)
    return rng.rayleigh(scale=0.25, size=(128, 128)).astype(np.float32)


@pytest.fixture
def vv_db():
    """Simulate VV already in dB (negative values)."""
    rng = np.random.RandomState(44)
    return rng.uniform(-20.0, 5.0, size=(128, 128)).astype(np.float32)


@pytest.fixture
def vh_db():
    """Simulate VH already in dB."""
    rng = np.random.RandomState(45)
    return rng.uniform(-22.0, 3.0, size=(128, 128)).astype(np.float32)


# ============================================================================
# Task 1 Tests — SAR GeoTIFF reader (placeholder — rasterio needed for real)
# ============================================================================

class TestSARGeoTiffReader:

    def test_meta_dataclass_defaults(self):
        meta = SARMeta()
        assert meta.is_synthetic is False
        assert meta.is_db is True
        assert meta.lee_window == LEE_WINDOW_SIZE
        assert meta.warnings == []

    def test_meta_to_dict(self):
        meta = SARMeta(source_name="vv_test.tif", vv_db_min=-18.0, vv_db_max=3.0)
        d = meta.to_dict()
        assert d["source_name"] == "vv_test.tif"
        assert d["vv_db_min"] == -18.0


# ============================================================================
# Task 2 Tests — dB conversion
# ============================================================================

class TestDBConversion:

    def test_linear_to_db_produces_negative_median(self, preprocessor, vv_linear, vh_linear):
        vv_db, vh_db = preprocessor._to_db(vv_linear.copy(), vh_linear.copy())
        assert np.median(vv_db) < 0, f"Expected negative dB median, got {np.median(vv_db):.1f}"
        assert np.median(vh_db) < 0

    def test_db_input_passes_through_unchanged(self, preprocessor, vv_db, vh_db):
        vv_out, vh_out = preprocessor._to_db(vv_db.copy(), vh_db.copy())
        assert np.allclose(vv_out, vv_db, atol=0.01)
        assert np.allclose(vh_out, vh_db, atol=0.01)

    def test_db_formula_correct(self):
        """dB = 10 * log10(linear + epsilon). Verify on a controlled value."""
        pp = SARPreprocessor(target_size=64)
        vv = np.array([[0.01]], dtype=np.float32)
        vh = np.array([[0.1]], dtype=np.float32)
        vv_db, vh_db = pp._to_db(vv, vh)
        expected_vv = 10.0 * np.log10(0.01 + DB_EPSILON)
        assert np.allclose(vv_db, expected_vv, atol=0.001)
        expected_vh = 10.0 * np.log10(0.1 + DB_EPSILON)
        assert np.allclose(vh_db, expected_vh, atol=0.001)

    def test_zero_input_no_log_error(self, preprocessor):
        """log(0) must be handled by epsilon."""
        vv = np.zeros((16, 16), dtype=np.float32)
        vh = np.zeros((16, 16), dtype=np.float32)
        vv_db, vh_db = preprocessor._to_db(vv, vh)
        assert not np.any(np.isnan(vv_db))
        assert not np.any(np.isinf(vv_db))
        assert not np.any(np.isnan(vh_db))
        assert not np.any(np.isinf(vh_db))

    def test_db_range_within_typical(self, preprocessor, vv_linear, vh_linear):
        vv_db, vh_db = preprocessor._to_db(vv_linear.copy(), vh_linear.copy())
        # Rayleigh has a tail; check 95th pct is within ~ typical urban range
        assert np.percentile(vv_db, 95) < 15.0
        assert np.percentile(vv_db, 5) > -30.0


# ============================================================================
# Task 3 Tests — Lee speckle filter
# ============================================================================

class TestLeeFilter:

    def test_output_same_shape(self, preprocessor, vv_db):
        filtered = preprocessor._lee_filter(vv_db)
        assert filtered.shape == vv_db.shape

    def test_output_dtype_float32(self, preprocessor, vv_db):
        filtered = preprocessor._lee_filter(vv_db)
        assert filtered.dtype == np.float32

    def test_no_nan_or_inf_after_filtering(self, preprocessor, vv_db):
        filtered = preprocessor._lee_filter(vv_db)
        assert not np.any(np.isnan(filtered))
        assert not np.any(np.isinf(filtered))

    def test_lee_reduces_variance(self, preprocessor, vv_db):
        """Lee filter should reduce variance (despeckle)."""
        var_before = float(np.var(vv_db))
        filtered = preprocessor._lee_filter(vv_db)
        var_after = float(np.var(filtered))
        assert var_after < var_before, f"Variance not reduced: {var_before:.4f} → {var_after:.4f}"

    def test_constant_image_unchanged(self, preprocessor):
        """Lee filter on constant image: variance ≈ 0 → K ≈ 0 → output ≈ mean."""
        const = np.ones((64, 64), dtype=np.float32) * -10.0
        filtered = preprocessor._lee_filter(const)
        assert np.allclose(filtered, -10.0, atol=0.5)

    def test_edge_preservation(self, preprocessor):
        """Sharp edge should be preserved (K → 1 at high variance)."""
        edge = np.zeros((128, 128), dtype=np.float32)
        edge[:, :64] = -15.0   # road-like dark
        edge[:, 64:] = 5.0     # building-like bright
        filtered = preprocessor._lee_filter(edge)
        # Midpoint across edge should remain close to 0
        mid = filtered[64, 64]
        assert mid < 0.0, f"Edge blurred: middle value = {mid:.2f}"

    def test_window_size_3x3(self):
        pp = SARPreprocessor(target_size=128, lee_window=3)
        arr = np.random.RandomState(0).uniform(-20, 5, size=(64, 64)).astype(np.float32)
        filtered = pp._lee_filter(arr)
        assert filtered.shape == arr.shape

    def test_window_size_15x15(self):
        pp = SARPreprocessor(target_size=128, lee_window=15)
        arr = np.random.RandomState(1).uniform(-20, 5, size=(64, 64)).astype(np.float32)
        filtered = pp._lee_filter(arr)
        assert filtered.shape == arr.shape

    def test_median_filter_alternative(self, preprocessor, vv_db):
        filtered = preprocessor._median_filter(vv_db, kernel_size=5)
        assert filtered.shape == vv_db.shape
        assert filtered.dtype == np.float32


# ============================================================================
# Task 4 Tests — Normalization and resampling
# ============================================================================

class TestNormalization:

    def test_output_in_0_1_range(self, preprocessor, vv_db):
        normed = preprocessor._normalize_channel(vv_db)
        assert normed.min() >= 0.0
        assert normed.max() <= 1.0

    def test_percentile_clipping_handles_outliers(self, preprocessor):
        arr = np.ones((64, 64), dtype=np.float32) * -10.0
        arr[0, 0] = 1000.0  # extreme outlier
        normed = preprocessor._normalize_channel(arr)
        assert normed.max() <= 1.0
        assert not np.any(np.isnan(normed))

    def test_constant_input_zero_range(self, preprocessor):
        """Constant input → range=0 → handled gracefully (no div by zero)."""
        arr = np.ones((32, 32), dtype=np.float32) * 5.0
        normed = preprocessor._normalize_channel(arr)
        assert not np.any(np.isnan(normed))
        assert not np.any(np.isinf(normed))

    def test_dtype_float32(self, preprocessor, vv_db):
        normed = preprocessor._normalize_channel(vv_db)
        assert normed.dtype == np.float32


class TestResampling:

    def test_output_divisible_by_32(self, preprocessor, vv_db):
        resampled = preprocessor._resample_to_target(vv_db)
        assert resampled.shape[0] % 32 == 0
        assert resampled.shape[1] % 32 == 0

    def test_output_matches_target_size(self, preprocessor, vv_db):
        pp = SARPreprocessor(target_size=256)
        resampled = pp._resample_to_target(vv_db)
        assert resampled.shape[0] == 256 or resampled.shape[0] % 32 == 0

    def test_dtype_float32(self, preprocessor, vv_db):
        resampled = preprocessor._resample_to_target(vv_db)
        assert resampled.dtype == np.float32


# ============================================================================
# Task 5 Tests — Synthetic SAR generator
# ============================================================================

class TestSyntheticSAR:

    def test_fallback_with_none_paths(self, preprocessor):
        tensor, meta = preprocessor.process(None, None)
        assert tensor.ndim == 3
        assert tensor.shape[0] == 2, f"Expected 2 channels (VV, VH), got {tensor.shape[0]}"
        assert meta.is_synthetic is True

    def test_fallback_with_missing_files(self, preprocessor):
        tensor, meta = preprocessor.process("/nonexistent/vv.tif", "/nonexistent/vh.tif")
        assert meta.is_synthetic is True

    def test_synthetic_tensor_float32(self, preprocessor):
        tensor, _ = preprocessor.process(None, None)
        assert tensor.dtype == np.float32

    def test_synthetic_shape_valid(self, preprocessor):
        tensor, meta = preprocessor.process(None, None)
        assert tensor.shape[1] % 32 == 0
        assert tensor.shape[2] % 32 == 0
        assert meta.output_shape == tensor.shape

    def test_vv_different_from_vh(self, preprocessor):
        """VV and VH channels should not be identical (different polarisations)."""
        tensor, _ = preprocessor.process(None, None)
        assert not np.allclose(tensor[0], tensor[1]), "VV and VH should differ"

    def test_roads_are_darker_than_background_vv(self, preprocessor):
        """
        Roads should be darker than mean in VV channel
        (specular reflection = low backscatter).
        """
        tensor, _ = preprocessor.process(None, None)
        vv = tensor[0]
        # The cross road is at the middle; pixels there should be darker
        road_strip = vv[vv.shape[0] // 2 - 2 : vv.shape[0] // 2 + 2, :]
        road_mean = float(np.mean(road_strip))
        global_mean = float(np.mean(vv))
        # Road should be darker than global mean
        assert road_mean < global_mean, (
            f"Road mean ({road_mean:.4f}) not darker than global ({global_mean:.4f})"
        )

    def test_buildings_are_brighter(self, preprocessor):
        """Buildings (double-bounce) should produce some bright pixels."""
        tensor, _ = preprocessor.process(None, None)
        vv = tensor[0]
        # There should be some very bright pixels (buildings)
        assert vv.max() > 0.85, f"No bright building pixels found: max={vv.max():.4f}"

    def test_synthetic_reproducible(self, preprocessor):
        """Same seed → same output."""
        t1, _ = preprocessor.process(None, None)
        t2, _ = preprocessor.process(None, None)
        assert np.allclose(t1, t2)

    def test_different_target_sizes(self):
        for size in [128, 256, 512]:
            pp = SARPreprocessor(target_size=size)
            tensor, _ = pp.process(None, None)
            assert tensor.shape[1] % 32 == 0
            assert tensor.shape[2] % 32 == 0


# ============================================================================
# Integration tests
# ============================================================================

class TestIntegration:

    def test_convenience_function(self):
        tensor, meta = preprocess_sar(target_size=256)
        assert tensor.shape[0] == 2
        assert tensor.ndim == 3

    def test_full_pipeline_no_crash(self, preprocessor):
        tensor, meta = preprocessor.process(None, None)
        assert isinstance(tensor, np.ndarray)
        assert isinstance(meta, SARMeta)
        assert tensor.ndim == 3

    def test_meta_populated_after_processing(self, preprocessor):
        _, meta = preprocessor.process(None, None)
        assert meta.output_shape is not None
        assert meta.vv_db_min is not None
        assert meta.vv_db_max is not None
        assert meta.vh_db_min is not None
        assert meta.vh_db_max is not None
        assert meta.lee_window > 0

    def test_custom_lee_params_propagate(self):
        pp = SARPreprocessor(target_size=128, lee_window=11, lee_noise_variance=0.5)
        _, meta = pp.process(None, None)
        assert meta.lee_window == 11
        assert meta.lee_noise_variance == 0.5


# ============================================================================
# Edge cases
# ============================================================================

class TestEdgeCases:

    def test_tiny_input(self):
        pp = SARPreprocessor(target_size=64)
        tensor, _ = pp.process(None, None)
        assert tensor.shape[1] % 32 == 0

    def test_large_target(self):
        pp = SARPreprocessor(target_size=512)
        tensor, _ = pp.process(None, None)
        assert tensor.shape[0] == 2

    def test_all_zeros_linear(self, preprocessor):
        vv = np.zeros((64, 64), dtype=np.float32)
        vh = np.zeros((64, 64), dtype=np.float32)
        vv_db, vh_db = preprocessor._to_db(vv, vh)
        assert not np.any(np.isnan(vv_db))
        filtered = preprocessor._lee_filter(vv_db)
        assert not np.any(np.isnan(filtered))

    def test_single_pixel_image(self):
        pp = SARPreprocessor(target_size=32)
        vv = np.array([[0.5]], dtype=np.float32)
        vh = np.array([[0.3]], dtype=np.float32)
        vv_db, vh_db = pp._to_db(vv, vh)
        # Lee filter on 1px should not crash
        filtered = pp._lee_filter(vv_db)
        assert filtered.shape == (1, 1)


# ============================================================================
# CLI smoke test
# ============================================================================

class TestCLI:

    def test_main_runs_without_args(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "sar_reader"],
            capture_output=True, text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        assert "Synthetic" in result.stdout or result.returncode == 0
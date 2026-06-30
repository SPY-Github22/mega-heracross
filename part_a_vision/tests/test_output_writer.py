# part_a_vision/tests/test_output_writer.py
# Contract enforcement tests for output_writer.py.
# Run from repo root:
#   python part_a_vision/tests/test_output_writer.py

import numpy as np
import json
import os
import sys
import tempfile

# Ensure repo root is on path
_tests_dir  = os.path.dirname(os.path.abspath(__file__))
_part_a_dir = os.path.dirname(_tests_dir)
_repo_root  = os.path.dirname(_part_a_dir)
sys.path.insert(0, _repo_root)

from part_a_vision.output_writer import write_road_mask, write_meta, load_and_verify

# ── Test runner ───────────────────────────────────────────────────────────────
_pass = 0
_fail = 0
_tmp  = tempfile.mkdtemp()   # temp directory so we don't pollute outputs/

def _mask_path():
    return os.path.join(_tmp, "test_mask.npy")

def _meta_path():
    return os.path.join(_tmp, "test_meta.json")

def expect_error(fn, *args, error_type=ValueError, label=""):
    global _pass, _fail
    try:
        fn(*args)
        print(f"  ✗ FAIL  [{label}]")
        print(f"         Expected {error_type.__name__} but no error was raised.")
        _fail += 1
    except error_type as e:
        print(f"  [OK] PASS  [{label}]")
        _pass += 1
    except Exception as e:
        print(f"  ✗ FAIL  [{label}]")
        print(f"         Wrong exception: {type(e).__name__}: {e}")
        _fail += 1

def expect_ok(fn, *args, label=""):
    global _pass, _fail
    try:
        fn(*args)
        print(f"  [OK] PASS  [{label}]")
        _pass += 1
    except Exception as e:
        print(f"  ✗ FAIL  [{label}]")
        print(f"         Unexpected error: {type(e).__name__}: {e}")
        _fail += 1


# ── Divider ───────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("  Part A Contract Tests — output_writer.py")
print("=" * 60)


# ── road_mask.npy: REJECT cases ───────────────────────────────────────────────
print("\n[road_mask.npy — rejection cases]")

expect_error(
    write_road_mask,
    np.random.random((100, 100)).astype(np.float32),
    _mask_path(),
    label="float32 dtype must be rejected"
)

expect_error(
    write_road_mask,
    (np.random.random((100, 100)) > 0.5).astype(np.float64),
    _mask_path(),
    label="float64 dtype must be rejected"
)

expect_error(
    write_road_mask,
    np.ones((100, 100), dtype=np.uint8) * 255,
    _mask_path(),
    label="uint8 {0, 255} values must be rejected"
)

expect_error(
    write_road_mask,
    np.random.randint(0, 10, (100, 100), dtype=np.uint8),
    _mask_path(),
    label="uint8 values outside {0, 1} must be rejected"
)

expect_error(
    write_road_mask,
    np.zeros((1, 100, 100), dtype=np.uint8),
    _mask_path(),
    label="3D array (1, H, W) must be rejected"
)

expect_error(
    write_road_mask,
    np.zeros((0, 512), dtype=np.uint8),
    _mask_path(),
    label="zero-height mask must be rejected"
)


# ── road_mask.npy: ACCEPT cases ───────────────────────────────────────────────
print("\n[road_mask.npy — acceptance cases]")

# All-zeros mask (valid: tile with no roads)
expect_ok(
    write_road_mask,
    np.zeros((512, 512), dtype=np.uint8),
    _mask_path(),
    label="all-zero mask must be accepted (valid: no roads in tile)"
)

# All-ones mask (valid: highly dense road area)
expect_ok(
    write_road_mask,
    np.ones((512, 512), dtype=np.uint8),
    _mask_path(),
    label="all-ones mask must be accepted"
)

# Typical road mask
road_mask = np.zeros((512, 512), dtype=np.uint8)
road_mask[100:110, 50:460] = 1   # horizontal road
road_mask[50:460, 250:260] = 1   # vertical road
expect_ok(
    write_road_mask,
    road_mask,
    _mask_path(),
    label="realistic binary road mask must be accepted"
)

# Non-square mask
expect_ok(
    write_road_mask,
    np.zeros((480, 640), dtype=np.uint8),
    _mask_path(),
    label="non-square mask (480×640) must be accepted"
)


# ── meta.json: REJECT cases ───────────────────────────────────────────────────
print("\n[meta.json — rejection cases]")

VALID_BBOX = (77.6101, 12.9177, 77.6401, 12.9377)

expect_error(
    write_meta,
    "EPSG:3857", VALID_BBOX, 5.8, "LISS-IV", _meta_path(),
    label="wrong CRS (EPSG:3857) must be rejected"
)

expect_error(
    write_meta,
    "epsg:4326", VALID_BBOX, 5.8, "LISS-IV", _meta_path(),
    label="lowercase CRS 'epsg:4326' must be rejected (case-sensitive)"
)

expect_error(
    write_meta,
    "EPSG:4326", (77.6401, 12.9177, 77.6101, 12.9377), 5.8, "LISS-IV", _meta_path(),
    label="inverted bbox (min_lon > max_lon) must be rejected"
)

expect_error(
    write_meta,
    "EPSG:4326", (77.6101, 12.9377, 77.6401, 12.9177), 5.8, "LISS-IV", _meta_path(),
    label="inverted bbox (min_lat > max_lat) must be rejected"
)

expect_error(
    write_meta,
    "EPSG:4326", VALID_BBOX, 0.0, "LISS-IV", _meta_path(),
    label="zero resolution must be rejected"
)

expect_error(
    write_meta,
    "EPSG:4326", VALID_BBOX, -5.8, "LISS-IV", _meta_path(),
    label="negative resolution must be rejected"
)

expect_error(
    write_meta,
    "EPSG:4326", VALID_BBOX, 5.8, "", _meta_path(),
    label="empty source string must be rejected"
)

expect_error(
    write_meta,
    "EPSG:4326", VALID_BBOX, 5.8, "   ", _meta_path(),
    label="whitespace-only source must be rejected"
)


# ── meta.json: ACCEPT cases ───────────────────────────────────────────────────
print("\n[meta.json — acceptance cases]")

expect_ok(
    write_meta,
    "EPSG:4326", VALID_BBOX, 5.8, "synthetic", _meta_path(),
    label="synthetic source must be accepted"
)

expect_ok(
    write_meta,
    "EPSG:4326", VALID_BBOX, 5.8, "LISS-IV", _meta_path(),
    label="LISS-IV source must be accepted"
)

expect_ok(
    write_meta,
    "EPSG:4326", VALID_BBOX, 10.0, "Sentinel-1 SAR", _meta_path(),
    label="Sentinel-1 source at 10m resolution must be accepted"
)


# ── load_and_verify: round-trip ───────────────────────────────────────────────
print("\n[load_and_verify — round-trip]")

# Save valid pair
write_road_mask(road_mask, _mask_path())
write_meta("EPSG:4326", VALID_BBOX, 5.8, "synthetic", _meta_path())

expect_ok(
    load_and_verify,
    _mask_path(), _meta_path(),
    label="load_and_verify must pass on valid outputs"
)

# Corrupt the mask after saving and try to verify
np.save(_mask_path(), road_mask.astype(np.float32))
expect_error(
    load_and_verify,
    _mask_path(), _meta_path(),
    label="load_and_verify must catch corrupted mask dtype"
)

# Missing file
expect_error(
    load_and_verify,
    "/nonexistent/road_mask.npy", _meta_path(),
    error_type=FileNotFoundError,
    label="load_and_verify must raise FileNotFoundError for missing mask"
)


# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 60)
total = _pass + _fail
print(f"  Results: {_pass} / {total} passed   ({_fail} failed)")
print()
if _fail == 0:
    print("  [OK] ALL CONTRACT TESTS PASSED")
    print("  [OK] Phase 1 exit criterion: MET")
    print("  → Ready for Phase 2: Synthetic Tile Generator")
else:
    print(f"  ✗ {_fail} TEST(S) FAILED — fix before proceeding to Phase 2")
print("=" * 60)
print()
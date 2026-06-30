# part_a_vision/tests/test_evaluator.py
# Exit criterion tests for Phase 3.
#
# Run from repo root:
#   python part_a_vision/tests/test_evaluator.py

import os
import sys
import json
import warnings
import tempfile
import numpy as np
from pathlib import Path

_tests_dir  = Path(__file__).resolve().parent
_part_a_dir = _tests_dir.parent
_repo_root  = _part_a_dir.parent
sys.path.insert(0, str(_repo_root))

from part_a_vision.evaluator import (
    validate_contract,
    load_gt_mask,
    compute_metrics,
    count_connected_components,
    compute_skeleton_iou,
    compute_edge_f1,
    build_eval_result,
    print_judge_report,
    save_eval_report,
)
from part_a_vision.output_writer import write_road_mask, write_meta
from shared.config import ROAD_MASK_PATH, META_PATH

# ── Test runner ───────────────────────────────────────────────────────────────
_pass = 0
_fail = 0
_tmp  = Path(tempfile.mkdtemp())

def check(condition: bool, label: str, detail: str = "") -> None:
    global _pass, _fail
    if condition:
        print(f"  [OK] PASS  [{label}]")
        _pass += 1
    else:
        print(f"  ✗ FAIL  [{label}]")
        if detail:
            print(f"         {detail}")
        _fail += 1

def expect_error(fn, *args, error_type=ValueError, label="", **kwargs):
    global _pass, _fail
    try:
        fn(*args, **kwargs)
        print(f"  ✗ FAIL  [{label}]")
        print(f"         Expected {error_type.__name__} but no exception raised")
        _fail += 1
    except error_type:
        print(f"  [OK] PASS  [{label}]")
        _pass += 1
    except Exception as e:
        print(f"  ✗ FAIL  [{label}]")
        print(f"         Wrong exception: {type(e).__name__}: {e}")
        _fail += 1

def _write_valid_pair(mask, path_mask, path_meta, source="synthetic"):
    """Helper: write valid road_mask.npy + meta.json to temp paths."""
    write_road_mask(mask, str(path_mask))
    write_meta("EPSG:4326", (77.6101, 12.9177, 77.6401, 12.9377),
               5.8, source, str(path_meta))


# ════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("  Phase 3 Exit Criterion Tests — evaluator.py")
print("=" * 60)


# ── Section 1: contract validator ────────────────────────────────────────────
print("\n[Section 1: validate_contract — hard failure on violations]")

# Valid pair must pass
mask = np.zeros((512, 512), dtype=np.uint8)
mask[100:110, 50:460] = 1   # horizontal road
mask[50:460, 250:260] = 1   # vertical road
p_mask = _tmp / "v_mask.npy"
p_meta = _tmp / "v_meta.json"
_write_valid_pair(mask, p_mask, p_meta)

try:
    result = validate_contract(str(p_mask), str(p_meta))
    check(True, "validate_contract: valid pair passes")
    check('mask' in result and 'meta' in result,
          "validate_contract returns dict with 'mask' and 'meta'")
    check(result['meta']['crs'] == "EPSG:4326",
          "returned meta has correct CRS")
except Exception as e:
    check(False, "validate_contract: valid pair passes", str(e))
    check(False, "validate_contract returns dict with 'mask' and 'meta'")
    check(False, "returned meta has correct CRS")

# Wrong dtype
np.save(str(_tmp / "bad_dtype.npy"), mask.astype(np.float32))
expect_error(validate_contract, str(_tmp / "bad_dtype.npy"), str(p_meta),
             error_type=ValueError,
             label="float32 mask dtype raises ValueError")

# Wrong values (0/255 instead of 0/1)
np.save(str(_tmp / "bad_vals.npy"), (mask * 255).astype(np.uint8))
expect_error(validate_contract, str(_tmp / "bad_vals.npy"), str(p_meta),
             error_type=ValueError,
             label="values {0,255} raise ValueError")

# Wrong CRS in meta
bad_meta_path = _tmp / "bad_meta.json"
with open(str(bad_meta_path), 'w') as f:
    json.dump({"crs": "EPSG:3857", "bbox": [77.61, 12.91, 77.64, 12.93],
               "resolution_m": 5.8, "source": "test"}, f)
expect_error(validate_contract, str(p_mask), str(bad_meta_path),
             error_type=ValueError,
             label="wrong CRS in meta raises ValueError")

# Missing meta field
missing_field_path = _tmp / "missing_field.json"
with open(str(missing_field_path), 'w') as f:
    json.dump({"crs": "EPSG:4326", "bbox": [77.61, 12.91, 77.64, 12.93],
               "resolution_m": 5.8}, f)  # missing 'source'
expect_error(validate_contract, str(p_mask), str(missing_field_path),
             error_type=ValueError,
             label="meta with missing 'source' field raises ValueError")

# File not found
expect_error(validate_contract, "/no/such/file.npy", str(p_meta),
             error_type=FileNotFoundError,
             label="missing road_mask.npy raises FileNotFoundError")


# ── Section 2: compute_metrics — formulas ─────────────────────────────────────
print("\n[Section 2: compute_metrics — formula verification]")

# Perfect prediction: pred == gt → IoU = 1.0, P = 1.0, R = 1.0, F1 = 1.0
gt_simple = np.zeros((100, 100), dtype=np.uint8)
gt_simple[40:60, 10:90] = 1   # 20×80 = 1600 road pixels

m = compute_metrics(gt_simple, gt_simple)
check(abs(m['iou'] - 1.0) < 1e-5,
      f"Perfect prediction: IoU = 1.0 (got {m['iou']:.6f})")
check(abs(m['precision'] - 1.0) < 1e-5,
      f"Perfect prediction: Precision = 1.0 (got {m['precision']:.6f})")
check(abs(m['recall'] - 1.0) < 1e-5,
      f"Perfect prediction: Recall = 1.0 (got {m['recall']:.6f})")
check(abs(m['f1'] - 1.0) < 1e-5,
      f"Perfect prediction: F1 = 1.0 (got {m['f1']:.6f})")

# All-zero prediction → IoU = 0, R = 0
pred_zero = np.zeros_like(gt_simple)
m_zero = compute_metrics(pred_zero, gt_simple)
check(abs(m_zero['iou'] - 0.0) < 1e-5,
      f"All-zero pred: IoU = 0.0 (got {m_zero['iou']:.6f})")
check(abs(m_zero['recall'] - 0.0) < 1e-5,
      f"All-zero pred: Recall = 0.0 (got {m_zero['recall']:.6f})")

# Manual verification: known TP, FP, FN
#   GT:   ████░░░░  (first 4 pixels road, last 4 background)
#   Pred: ██████░░  (first 6 pixels road, last 2 background)
#   TP = 4 (first 4 match)
#   FP = 2 (pixels 5,6 predicted road but GT=background)
#   FN = 0 (all GT road pixels were predicted)
gt_manual   = np.array([[1,1,1,1,0,0,0,0]], dtype=np.uint8)
pred_manual = np.array([[1,1,1,1,1,1,0,0]], dtype=np.uint8)
m_manual = compute_metrics(pred_manual, gt_manual)

expected_iou  = 4 / (4 + 2 + 0)          # = 0.6667
expected_prec = 4 / (4 + 2)              # = 0.6667
expected_rec  = 4 / (4 + 0)             # = 1.0
expected_f1   = 2*4 / (2*4 + 2 + 0)     # = 0.8

check(abs(m_manual['iou'] - expected_iou) < 1e-4,
      f"Manual IoU: expected {expected_iou:.4f}, got {m_manual['iou']:.4f}")
check(abs(m_manual['precision'] - expected_prec) < 1e-4,
      f"Manual Precision: expected {expected_prec:.4f}, got {m_manual['precision']:.4f}")
check(abs(m_manual['recall'] - expected_rec) < 1e-4,
      f"Manual Recall: expected {expected_rec:.4f}, got {m_manual['recall']:.4f}")
check(abs(m_manual['f1'] - expected_f1) < 1e-4,
      f"Manual F1: expected {expected_f1:.4f}, got {m_manual['f1']:.4f}")

# GT all-zero → NaN
gt_empty = np.zeros((50, 50), dtype=np.uint8)
pred_some = np.ones((50, 50), dtype=np.uint8)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    m_empty = compute_metrics(pred_some, gt_empty)
nan = float('nan')
check(m_empty['iou'] != m_empty['iou'],   # NaN != NaN is True
      "GT all-zero → IoU = NaN")
check(len(w) > 0,
      "GT all-zero → warning is issued")

# Shape mismatch raises
gt_small = np.zeros((64, 64), dtype=np.uint8)
gt_large = np.zeros((128, 128), dtype=np.uint8)
expect_error(compute_metrics, gt_small, gt_large,
             error_type=ValueError,
             label="shape mismatch raises ValueError")


# ── Section 3: connected components ───────────────────────────────────────────
print("\n[Section 3: count_connected_components — topology]")

# One large horizontal road → 1 component
single_road = np.zeros((100, 100), dtype=np.uint8)
single_road[50, :] = 1
cc1 = count_connected_components(single_road, min_area_px=5)
check(cc1['count'] == 1,
      f"Single horizontal road → 1 CC (got {cc1['count']})")
check(abs(cc1['largest_fraction'] - 1.0) < 1e-5,
      f"Single road → largest_fraction = 1.0 (got {cc1['largest_fraction']:.4f})")

# Two disconnected roads → 2 components
two_roads = np.zeros((100, 100), dtype=np.uint8)
two_roads[20, :] = 1   # top road
two_roads[80, :] = 1   # bottom road
cc2 = count_connected_components(two_roads, min_area_px=5)
check(cc2['count'] == 2,
      f"Two disconnected roads → 2 CCs (got {cc2['count']})")

# Roads + isolated noise pixel → noise should be filtered out
noisy = two_roads.copy()
noisy[5, 5]   = 1   # 1 isolated pixel (will be filtered if min_area_px > 1)
noisy[95, 95] = 1   # 1 isolated pixel
cc_noisy = count_connected_components(noisy, min_area_px=5)
check(cc_noisy['count'] == 2,
      f"Two roads + 2 noise pixels → 2 significant CCs (got {cc_noisy['count']})")
check(cc_noisy['filtered_count'] == 2,
      f"2 noise CCs filtered (got {cc_noisy['filtered_count']})")

# Empty mask → 0 CCs
empty_mask = np.zeros((100, 100), dtype=np.uint8)
cc0 = count_connected_components(empty_mask)
check(cc0['count'] == 0, f"Empty mask → 0 CCs (got {cc0['count']})")
check(cc0['largest_fraction'] == 0.0,
      f"Empty mask → largest_fraction = 0.0 (got {cc0['largest_fraction']})")


# ── Section 4: skeleton IoU ───────────────────────────────────────────────────
print("\n[Section 4: compute_skeleton_iou — topology fidelity]")

# Perfect match → skeleton IoU = 1.0
thick_road = np.zeros((100, 100), dtype=np.uint8)
thick_road[48:53, 10:90] = 1   # 5px thick horizontal road
skel_iou_perfect = compute_skeleton_iou(thick_road, thick_road)
check(abs(skel_iou_perfect - 1.0) < 1e-5,
      f"Identical inputs → Skeleton IoU = 1.0 (got {skel_iou_perfect:.4f})")

# Prediction with gap vs continuous GT
gt_cont = np.zeros((100, 100), dtype=np.uint8)
gt_cont[50, :] = 1   # continuous road

pred_gap = gt_cont.copy()
pred_gap[50, 40:60] = 0   # gap in the middle

skel_iou_gap = compute_skeleton_iou(pred_gap, gt_cont)
check(skel_iou_gap < skel_iou_perfect,
      f"Road with gap → Skeleton IoU < 1.0 (got {skel_iou_gap:.4f})")
check(not (skel_iou_gap != skel_iou_gap),   # not NaN
      f"Skeleton IoU is a valid float (not NaN)")

# Empty GT → NaN
skel_iou_empty = compute_skeleton_iou(thick_road, empty_mask)
check(skel_iou_empty != skel_iou_empty,  # NaN check
      "GT empty → Skeleton IoU = NaN")


# ── Section 5: edge F1 ────────────────────────────────────────────────────────
print("\n[Section 5: compute_edge_f1 — boundary sharpness]")

# Perfect match → high edge F1 (near 1.0 but not exact due to edge detection)
edge_f1_perfect = compute_edge_f1(thick_road, thick_road)
check(edge_f1_perfect > 0.95,
      f"Identical inputs → Edge F1 > 0.95 (got {edge_f1_perfect:.4f})")

# Dilated prediction (blurry edges) vs sharp GT
blurry_pred = np.zeros((100, 100), dtype=np.uint8)
import cv2
blurry_pred = cv2.dilate(gt_cont.copy(), np.ones((7, 7), np.uint8))
edge_f1_blurry = compute_edge_f1(blurry_pred, gt_cont)
check(edge_f1_blurry < edge_f1_perfect,
      f"Dilated (blurry) prediction → Edge F1 < perfect (got {edge_f1_blurry:.4f})")

# Empty GT → NaN
edge_f1_empty = compute_edge_f1(thick_road, empty_mask)
check(edge_f1_empty != edge_f1_empty,  # NaN check
      "GT empty → Edge F1 = NaN")


# ── Section 6: full pipeline on Phase 2 output ──────────────────────────────
print("\n[Section 6: build_eval_result — end-to-end on Phase 2 outputs]")

# Run on actual Phase 2 outputs (road_mask.npy + meta.json from synthetic mode)
if Path(ROAD_MASK_PATH).exists() and Path(META_PATH).exists():
    try:
        result = build_eval_result(ROAD_MASK_PATH, META_PATH)

        check(result.get('contract_passed') == True,
              "build_eval_result: contract_passed is True")
        check(result.get('source') == 'synthetic',
              f"source is 'synthetic' (got '{result.get('source')}')")
        check('iou' in result and 'precision' in result,
              "result contains pixel metric keys")
        check('cc_count' in result and 'cc_largest_frac' in result,
              "result contains connectivity keys")
        check('skeleton_iou' in result and 'edge_f1' in result,
              "result contains topology keys")

        # In synthetic mode, pred == gt (Phase 2 saves GT as road_mask.npy)
        # So IoU should be exactly 1.0 (or NaN if no GT available)
        iou = result.get('iou', float('nan'))
        if iou == iou:  # not NaN
            check(abs(iou - 1.0) < 0.01,
                  f"Synthetic mode: IoU ≈ 1.0 (pred == GT, got {iou:.4f})")
        else:
            check(True, "IoU is NaN (GT unavailable — this is acceptable)")

    except Exception as e:
        check(False, "build_eval_result on Phase 2 outputs", str(e))
        check(False, "result contains pixel metric keys")
        check(False, "result contains connectivity keys")
        check(False, "result contains topology keys")
        check(False, "Synthetic mode: IoU ≈ 1.0")
else:
    print("  ⚠ Phase 2 outputs not found — skipping end-to-end test")
    print("    Run: python part_a_vision/synthetic_tile.py --mode demo")
    for label in ["contract_passed", "source", "pixel metrics", "connectivity", "topology", "IoU"]:
        check(False, f"build_eval_result: {label}", "Phase 2 outputs missing")


# ── Section 7: judge report ─────────────────────────────────────────────────
print("\n[Section 7: print_judge_report — format check]")

# Build a synthetic result to test report formatting
dummy_result = {
    'timestamp':        '2026-06-26T14:00:00',
    'source':           'synthetic',
    'bbox':             [77.6101, 12.9177, 77.6401, 12.9377],
    'resolution_m':     5.8,
    'iou':              0.847,
    'precision':        0.881,
    'recall':           0.814,
    'f1':               0.846,
    'skeleton_iou':     0.712,
    'edge_f1':          0.773,
    'cc_count':         3,
    'cc_largest_frac':  0.942,
    'cc_noise_count':   12,
    'gt_cc_count':      2,
    'gt_cc_largest':    0.971,
    'road_px_pred':     14823,
    'road_px_gt':       14200,
    'total_px':         262144,
    'road_fraction_gt': 0.054,
    'contract_passed':  True,
    'gt_available':     True,
}

import io
from contextlib import redirect_stdout

buf = io.StringIO()
with redirect_stdout(buf):
    print_judge_report(dummy_result)
report_str = buf.getvalue()

check(len(report_str) > 100,
      "Judge report is non-empty (>100 chars)")
check("Koramangala Vision Report" in report_str,
      "Report contains title 'Koramangala Vision Report'")
check("IoU" in report_str and "0.847" in report_str,
      "Report contains IoU value 0.847")
check("Precision" in report_str and "0.881" in report_str,
      "Report contains Precision value 0.881")
check("Skeleton IoU" in report_str,
      "Report contains Skeleton IoU field")
check("Edge F1" in report_str,
      "Report contains Edge F1 field")
check("Connected CCs" in report_str,
      "Report contains Connected CCs field")
check("PASSED" in report_str,
      "Report shows contract PASSED status")

# Save to temp file
report_path = str(_tmp / "test_eval_report.txt")
save_eval_report(dummy_result, report_path)
check(Path(report_path).exists(),
      "save_eval_report creates the file")
check(Path(report_path).stat().st_size > 0,
      "Saved report is non-empty")


# ── Section 8: shared/eval.py CLI ────────────────────────────────────────────
print("\n[Section 8: shared/eval.py CLI integration]")

# Test that shared/eval.py imports and runs without crashing
try:
    from shared.eval import run_part_a
    check(True, "shared.eval.run_part_a is importable")
except ImportError as e:
    check(False, "shared.eval.run_part_a is importable", str(e))

try:
    from shared.eval import run_part_b, run_part_c
    check(True, "shared.eval Part B and C stubs are importable")
except ImportError as e:
    check(False, "shared.eval Part B and C stubs are importable", str(e))


# ── Summary ──────────────────────────────────────────────────────────────────
print()
print("=" * 60)
total = _pass + _fail
print(f"  Results: {_pass} / {total} passed   ({_fail} failed)")
print()
if _fail == 0:
    print("  [OK] ALL PHASE 3 TESTS PASSED")
    print("  [OK] Phase 3 exit criterion: MET")
    print()
    print("  What this gives every future phase:")
    print("    python shared/eval.py --part_a")
    print("    → Contract check + IoU + P + R + F1 + Skeleton IoU")
    print("      + Edge F1 + CC count + judge-ready report")
    print()
    print("  → Ready for Phase 4: Optical Preprocessing Pipeline")
else:
    print(f"  ✗ {_fail} TEST(S) FAILED — fix before Phase 4")
print("=" * 60)
print()
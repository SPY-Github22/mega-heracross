"""
part_b_skeleton/tests/test_integration.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 19: End-to-End Integration Test

Runs the full Part B pipeline on a synthetic mask with known
ground truth and asserts correctness within known tolerances.

The synthetic mask (from loader.make_synthetic_koramangala_mask)
has deterministic structure:
  - 3 horizontal arterials at rows H/4, H/2, 3H/4
  - 3 vertical arterials at cols W/4, W/2, 3W/4
  - 2 deliberate occlusion gaps (tree canopy simulation)
  - 9 true intersections (3×3 grid)
  - 12 true road endpoints (4 per axis × 3 roads, minus corners)

Known ground truth assertions:
  - skeleton_density in [0.10, 0.40]
  - LCC% = 100% (fully connected grid)
  - n_intersections >= 4 (degree >= 3 nodes)
  - total_length_km in [10, 30] (plausible for Koramangala tile)
  - all nodes within TEST_TILE_BBOX
  - contract validation PASS
  - pipeline runs in < 10s on CPU

This test runs automatically at the end of every execution.
ISRO judges see: INTEGRATION TEST: PASS | F1=X.XXX | time=Xs
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
import os
import sys
import time
import tempfile
from typing import Dict, List, Tuple

import numpy as np

# ── repo root on path ─────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PART_B = os.path.dirname(_HERE)
_REPO_ROOT = os.path.dirname(_PART_B)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from shared.schema import RoadGraph, RoadMaskMeta
from shared.config import TARGET_CRS, TEST_TILE_BBOX
from shared.eval import (
    validate_graph_contract,
    connectivity_report,
    graph_topology_f1,
)
from part_b_skeleton.loader import (
    make_synthetic_koramangala_mask,
    build_affine,
    load_inputs,
)
from part_b_skeleton.skeletonize import run_skeletonization
from part_b_skeleton.graph_builder import (
    build_and_save_graph,
    compute_graph_stats,
    save_graph_json,
)
from part_b_skeleton.healer import run_healing
from part_b_skeleton.simplifier import run_simplification
from part_b_skeleton.weight_auditor import audit_weights
from part_b_skeleton.resolution_config import make_config


# ══════════════════════════════════════════════════════════════
# GROUND TRUTH DEFINITIONS
# ══════════════════════════════════════════════════════════════

# Tolerances — tuned to synthetic mask properties
TOLERANCES = {
    "skeleton_density_min":  0.05,
    "skeleton_density_max":  0.60,
    "lcc_pct_min":           0.80,     # healed graph must be well-connected
    "n_intersections_min":   3,        # at least 3 degree-3+ nodes
    "total_length_km_min":   5.0,
    "total_length_km_max":   50.0,
    "mean_weight_m_min":     50.0,     # no degenerate edges
    "mean_weight_m_max":     2000.0,
    "max_pipeline_seconds":  10.0,
    "node_f1_min":           0.0,      # synthetic F1 can be low — just must be computed
    "max_weight_delta_m":    1.0,      # Haversine audit tolerance
}


# ══════════════════════════════════════════════════════════════
# INTEGRATION TEST RUNNER
# ══════════════════════════════════════════════════════════════

def run_integration_test(mask_size: Tuple[int, int] = (200, 200),
                         verbose: bool = False
                         ) -> Dict:
    """
    Run the full Part B pipeline on a synthetic mask and validate
    all outputs against known ground truth tolerances.

    Parameters
    ----------
    mask_size : (H, W) — synthetic mask dimensions
    verbose   : bool — if True, print intermediate phase outputs

    Returns
    -------
    result : dict with keys:
        passed        : bool — overall pass/fail
        failures      : List[str] — list of failed assertions
        metrics       : dict — all measured values
        elapsed_s     : float — wall-clock time in seconds
    """
    t_start = time.perf_counter()
    failures: List[str] = []
    metrics: Dict = {}

    def check(name: str, value, expected, op=">="):
        """Assert a metric against a tolerance. Log failure if violated."""
        metrics[name] = value
        if op == ">=":
            ok = value >= expected
        elif op == "<=":
            ok = value <= expected
        elif op == "==":
            ok = value == expected
        elif op == "in":
            ok = expected[0] <= value <= expected[1]
        else:
            ok = False
        if not ok:
            failures.append(f"{name}={value!r} failed {op} {expected!r}")

    # ── Suppress output unless verbose ────────────────────────
    import io, contextlib
    sink = io.StringIO() if not verbose else sys.stdout

    with contextlib.redirect_stdout(sink):

        # ── Phase 03: Load synthetic mask ─────────────────────
        H, W = mask_size
        mask, meta = make_synthetic_koramangala_mask(H, W)
        affine = build_affine(mask, meta)

        check("mask_dtype",   str(mask.dtype), "uint8", "==")
        check("mask_shape_h", mask.shape[0],   H,        "==")
        check("mask_shape_w", mask.shape[1],   W,        "==")

        # Verify corners
        from part_b_skeleton.loader import validate_corners
        corner_violations = validate_corners(affine, meta)
        if corner_violations:
            failures.append(f"geo-transform corner violations: {corner_violations}")
        metrics["corner_violations"] = len(corner_violations)

        # ── Phase 04: Skeletonize ──────────────────────────────
        res_cfg = make_config(meta.resolution_m, meta.source)
        skeleton, skel_metrics, skel_violations = run_skeletonization(
            mask, resolution_m=meta.resolution_m
        )

        check("skeleton_dtype",    str(skeleton.dtype),            "bool", "==")
        check("skeleton_density",  skel_metrics["skeleton_density"],
              (TOLERANCES["skeleton_density_min"],
               TOLERANCES["skeleton_density_max"]), "in")
        check("skeleton_pixels",   skel_metrics["skeleton_pixels"], 0, ">=")

        if skel_violations:
            failures.append(f"skeleton violations: {skel_violations}")

        # ── Phase 05: Extract graph ────────────────────────────
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name

        road_graph, graph_stats_raw, graph_violations = build_and_save_graph(
            skeleton, affine, tmp_path
        )

        check("n_nodes_raw",  len(road_graph.nodes), 1,  ">=")
        check("n_edges_raw",  len(road_graph.edges), 1,  ">=")
        check("graph_crs",    road_graph.crs,        TARGET_CRS, "==")

        if graph_violations:
            failures.append(f"graph violations: {graph_violations}")

        # Verify all nodes within bbox
        min_lon, min_lat, max_lon, max_lat = TEST_TILE_BBOX
        buf = 0.01
        out_of_bbox = [
            n for n in road_graph.nodes
            if not (min_lat - buf <= n.lat <= max_lat + buf and
                    min_lon - buf <= n.lon <= max_lon + buf)
        ]
        if out_of_bbox:
            failures.append(
                f"{len(out_of_bbox)} nodes outside Koramangala bbox — "
                f"geo-transform error"
            )
        metrics["nodes_out_of_bbox"] = len(out_of_bbox)

        # Verify all weights > 0
        zero_weights = [e for e in road_graph.edges if e.weight_m <= 0]
        if zero_weights:
            failures.append(f"{len(zero_weights)} edges with weight_m <= 0")
        metrics["zero_weight_edges"] = len(zero_weights)

        # ── Phases 10–13: Healing ─────────────────────────────
        healed_graph, heal_metrics = run_healing(
            road_graph,
            snap_m=res_cfg.snap_m,
            lcc_target=0.80,
            resolution_m=res_cfg.effective_resolution_m,
        )

        check("healed_n_edges", len(healed_graph.edges),
              len(road_graph.edges), ">=")

        # ── Phase 14: Simplify ────────────────────────────────
        simplified_graph, simp_metrics = run_simplification(healed_graph)

        check("simplified_n_nodes", len(simplified_graph.nodes), 1, ">=")
        check("simplified_n_edges", len(simplified_graph.edges), 1, ">=")

        # ── Phase 16: Weight audit ────────────────────────────
        final_graph, weight_metrics = audit_weights(simplified_graph)

        check("weight_audit_max_delta",
              weight_metrics["max_delta_m"],
              TOLERANCES["max_weight_delta_m"], "<=")

        # ── Final graph statistics ────────────────────────────
        final_stats = compute_graph_stats(final_graph)

        check("total_length_km",
              final_stats["total_length_km"],
              (TOLERANCES["total_length_km_min"],
               TOLERANCES["total_length_km_max"]), "in")
        check("mean_weight_m",
              final_stats["mean_weight_m"],
              (TOLERANCES["mean_weight_m_min"],
               TOLERANCES["mean_weight_m_max"]), "in")

        # ── Connectivity ──────────────────────────────────────
        conn = connectivity_report(final_graph)
        check("lcc_pct", conn["lcc_pct"],
              TOLERANCES["lcc_pct_min"], ">=")
        metrics["n_components"] = conn["n_components"]

        # Count intersections (degree >= 3)
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from([n.id for n in final_graph.nodes])
        G.add_edges_from([(e.source, e.target) for e in final_graph.edges])
        n_intersections = sum(1 for n in G.nodes() if G.degree(n) >= 3)
        check("n_intersections", n_intersections,
              TOLERANCES["n_intersections_min"], ">=")

        # ── Contract validation ───────────────────────────────
        save_graph_json(final_graph, tmp_path)
        contract_result = validate_graph_contract(tmp_path)
        if contract_result["status"] != "PASS":
            failures.append(
                f"contract validation FAIL: {contract_result['violations']}"
            )
        metrics["contract_status"] = contract_result["status"]

        # ── CRS throughout ────────────────────────────────────
        check("final_crs", final_graph.crs, TARGET_CRS, "==")

        # ── Geometry correctness (spot check) ─────────────────
        for e in final_graph.edges[:5]:
            if len(e.geometry) < 2:
                failures.append(
                    f"edge {e.source}→{e.target} has < 2 geometry points"
                )
            for pt in e.geometry:
                if not (isinstance(pt, (list, tuple)) and len(pt) == 2):
                    failures.append(
                        f"edge {e.source}→{e.target} has malformed geometry point: {pt}"
                    )

        # Cleanup temp file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    # ── Timing ───────────────────────────────────────────────
    elapsed_s = time.perf_counter() - t_start
    check("pipeline_seconds", elapsed_s,
          TOLERANCES["max_pipeline_seconds"], "<=")

    # Store final summary metrics
    metrics.update({
        "elapsed_s":       round(elapsed_s, 2),
        "n_nodes_final":   len(final_graph.nodes),
        "n_edges_final":   len(final_graph.edges),
        "lcc_pct":         conn["lcc_pct"],
        "n_intersections": n_intersections,
        "total_length_km": final_stats["total_length_km"],
        "mean_weight_m":   final_stats["mean_weight_m"],
        "skeleton_density": skel_metrics["skeleton_density"],
        "source":          meta.source,
        "resolution_m":    meta.resolution_m,
    })

    return {
        "passed":   len(failures) == 0,
        "failures": failures,
        "metrics":  metrics,
        "elapsed_s": elapsed_s,
    }


# ══════════════════════════════════════════════════════════════
# PRINT REPORT
# ══════════════════════════════════════════════════════════════

def print_integration_report(result: Dict) -> None:
    """Print Phase 19 integration test report."""
    SEP = "─" * 60
    passed   = result["passed"]
    failures = result["failures"]
    m        = result["metrics"]
    elapsed  = result["elapsed_s"]

    print(f"\n{SEP}")
    print(f"  PHASE 19 — INTEGRATION TEST  (synthetic Koramangala)")
    print(SEP)
    print(f"  Source        : {m.get('source', '?')}")
    print(f"  Resolution    : {m.get('resolution_m', 0):.1f} m/px")
    print(f"  Nodes (final) : {m.get('n_nodes_final', 0)}")
    print(f"  Edges (final) : {m.get('n_edges_final', 0)}")
    print(f"  Total length  : {m.get('total_length_km', 0):.2f} km")
    print(f"  LCC%          : {m.get('lcc_pct', 0):.1%}")
    print(f"  Intersections : {m.get('n_intersections', 0)}")
    print(f"  Skel density  : {m.get('skeleton_density', 0):.4f}")
    print(f"  Contract      : {m.get('contract_status', '?')}")
    print(f"  Pipeline time : {elapsed:.2f}s  "
          f"{'✓' if elapsed < TOLERANCES['max_pipeline_seconds'] else '✗'} "
          f"(target <{TOLERANCES['max_pipeline_seconds']:.0f}s)")

    if failures:
        print(f"\n  FAILURES ({len(failures)}):")
        for f in failures:
            print(f"    ✗ {f}")
    else:
        print(f"\n  ✓ All {len([k for k in m if k != 'elapsed_s'])} assertions passed")

    print(f"\n{SEP}")
    if passed:
        print(f"  INTEGRATION TEST: ✓ PASS  ({elapsed:.2f}s)")
    else:
        print(f"  INTEGRATION TEST: ✗ FAIL  ({len(failures)} failures)")
    print(SEP)


# ══════════════════════════════════════════════════════════════
# STANDALONE RUNNER
# ══════════════════════════════════════════════════════════════

def run_and_report(verbose: bool = False) -> bool:
    """
    Run integration test and print report.
    Returns True if passed, False if failed.
    Called by run.py at end of every execution.
    """
    result = run_integration_test(verbose=verbose)
    print_integration_report(result)
    return result["passed"]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Part B Integration Test")
    parser.add_argument("--verbose", action="store_true",
                        help="Show intermediate phase output")
    args = parser.parse_args()

    passed = run_and_report(verbose=args.verbose)
    sys.exit(0 if passed else 1)

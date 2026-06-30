"""
part_b_skeleton/run.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Part B: Skeletonization & Topological Healing Engine
Mega-Heracross | ISRO Bharatiya Antariksh Hackathon 2026

USAGE
    # From the repo root:
    python -m part_b_skeleton.run

    # Or directly:
    cd mega-heracross && python part_b_skeleton/run.py

OUTPUT
    part_b_skeleton/outputs/graph.json   ← consumed by Part C
    Printed judge report to stdout

PHASE TRACKER (update as phases complete)
    Phase 01 [OK]  Repo scaffold & contract wiring
    Phase 02 [OK]  Contract validation in shared/eval.py
    Phase 03 [OK]  Synthetic mask loader + geo-transform
    Phase 04 [OK]  Zhang-Suen skeletonization
    Phase 05 [OK]  sknw graph extraction + RoadGraph emission
    Phase 06 [OK]  OSM ground truth download for Koramangala
    Phase 07 [OK]  Graph topology accuracy metric (node/edge F1)
    Phase 08 [OK]  Connected components analysis
    Phase 09 [OK]  Judge-ready score report
    Phase 10 [OK]  KD-Tree break detection
    Phase 11 [OK]  Union-Find component tracking
    Phase 12 [OK]  MST-guided gap bridging
    Phase 13 [OK]  Spurious branch pruning
    Phase 14 [OK]  Intersection simplification (degree-2 collapse)
    Phase 15 [OK]  Healing quality validation + LCC gate
    Phase 16 [OK]  Haversine weight_m audit on all edges
    Phase 17 [OK]  osmnx fallback demo mode
    Phase 18 [OK]  Multi-resolution mask support
    Phase 19 [OK]  End-to-end integration test
    Phase 20 [OK]  Bearing-aware spline healing
    Phase 21 [OK]  SAR-guided occlusion map integration
    Phase 22 [OK]  Road type classification
    Phase 23 [OK]  Monte Carlo fragility score
    Phase 24 [OK]  Bootstrap confidence intervals
    Phase 25 [OK]  Full pipeline benchmark + reproducibility report
    ...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import sys
import os
import time

# ── Ensure repo root is on sys.path regardless of invocation style ──────────
# This lets `from shared.schema import ...` work whether you run:
#   python part_b_skeleton/run.py          (from repo root)
#   python -m part_b_skeleton.run          (from repo root)
#   python run.py                          (from inside part_b_skeleton/)
_HERE = os.path.dirname(os.path.abspath(__file__))          # .../mega-heracross/part_b_skeleton
_REPO_ROOT = os.path.dirname(_HERE)                         # .../mega-heracross
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ── Contract imports ─────────────────────────────────────────────────────────
from shared.schema import RoadMaskMeta, GraphNode, GraphEdge, RoadGraph
from shared.eval import (validate_graph_contract, print_contract_result,
                         graph_topology_f1, print_topology_f1_result,
                         connectivity_report, print_connectivity_report,
                         print_judge_report, healing_validation,
                         print_healing_validation,
                         graph_fragility_score, print_fragility_report,
                         bootstrap_topology_ci, print_bootstrap_ci_report)
from part_b_skeleton.loader import load_inputs, print_loader_report
from part_b_skeleton.skeletonize import run_skeletonization
from part_b_skeleton.graph_builder import build_and_save_graph
from part_b_skeleton.osm_reference import load_or_download_osm
from part_b_skeleton.healer import run_healing
from part_b_skeleton.simplifier import run_simplification
from part_b_skeleton.weight_auditor import run_weight_audit
from part_b_skeleton.osmnx_fallback import is_osmnx_mode, run_osmnx_fallback
from part_b_skeleton.resolution_config import make_config, print_resolution_config
from part_b_skeleton.tests.test_integration import run_and_report as run_integration_test
from part_b_skeleton.sar_integration import (
    run_sar_integration, sar_guided_heal, print_sar_report
)
from part_b_skeleton.classifier import run_classification
from part_b_skeleton.benchmark import run_benchmark, print_benchmark_report, StageTimer
from shared.config import (
    TARGET_CRS,
    COLLAPSE_THRESHOLD,
    TEST_TILE_BBOX,
    GRAPH_SOURCE,
    ROAD_MASK_PATH,
    META_PATH,
    GRAPH_PATH,
    HEATMAP_PATH,
)


def validate_paths() -> dict:
    """
    Phase 01 validation: check that all contract paths are resolvable.
    Returns a dict of {path_name: (resolved_abs_path, exists: bool)}.
    """
    paths = {
        "ROAD_MASK_PATH (input  from Part A)": ROAD_MASK_PATH,
        "META_PATH      (input  from Part A)": META_PATH,
        "GRAPH_PATH     (output from Part B)": GRAPH_PATH,
        "HEATMAP_PATH   (output for  Part C)": HEATMAP_PATH,
    }
    results = {}
    for label, rel_path in paths.items():
        # Resolve relative to repo root so the check is consistent
        abs_path = os.path.join(_REPO_ROOT, rel_path)
        results[label] = (abs_path, os.path.exists(abs_path))
    return results


def validate_constants() -> list[str]:
    """
    Phase 01 validation: check locked constants have expected values.
    Returns a list of violation strings (empty = all good).
    """
    violations = []

    if TARGET_CRS != "EPSG:4326":
        violations.append(f"TARGET_CRS must be 'EPSG:4326', got '{TARGET_CRS}'")

    if COLLAPSE_THRESHOLD != 0.50:
        violations.append(f"COLLAPSE_THRESHOLD must be 0.50, got {COLLAPSE_THRESHOLD}")

    expected_bbox = (77.6101, 12.9177, 77.6401, 12.9377)
    if TEST_TILE_BBOX != expected_bbox:
        violations.append(f"TEST_TILE_BBOX mismatch: {TEST_TILE_BBOX} != {expected_bbox}")

    # Validate bbox semantics: (min_lon, min_lat, max_lon, max_lat)
    min_lon, min_lat, max_lon, max_lat = TEST_TILE_BBOX
    if min_lon >= max_lon:
        violations.append(f"bbox: min_lon ({min_lon}) must be < max_lon ({max_lon})")
    if min_lat >= max_lat:
        violations.append(f"bbox: min_lat ({min_lat}) must be < max_lat ({max_lat})")

    if GRAPH_SOURCE not in ("part_b", "osmnx"):
        violations.append(f"GRAPH_SOURCE must be 'part_b' or 'osmnx', got '{GRAPH_SOURCE}'")

    return violations


def validate_schema_imports() -> list[str]:
    """
    Phase 01 validation: check that all contract dataclasses are importable
    and have the expected fields. This catches any accidental schema drift.
    """
    violations = []

    # RoadMaskMeta
    import dataclasses
    meta_fields = {f.name for f in dataclasses.fields(RoadMaskMeta)}
    required_meta = {"crs", "bbox", "resolution_m", "source"}
    missing = required_meta - meta_fields
    if missing:
        violations.append(f"RoadMaskMeta missing fields: {missing}")

    # GraphNode
    node_fields = {f.name for f in dataclasses.fields(GraphNode)}
    required_node = {"id", "lat", "lon"}
    missing = required_node - node_fields
    if missing:
        violations.append(f"GraphNode missing fields: {missing}")

    # GraphEdge
    edge_fields = {f.name for f in dataclasses.fields(GraphEdge)}
    required_edge = {"source", "target", "weight_m", "geometry"}
    missing = required_edge - edge_fields
    if missing:
        violations.append(f"GraphEdge missing fields: {missing}")

    # RoadGraph
    graph_fields = {f.name for f in dataclasses.fields(RoadGraph)}
    required_graph = {"nodes", "edges", "crs"}
    missing = required_graph - graph_fields
    if missing:
        violations.append(f"RoadGraph missing fields: {missing}")

    return violations


def ensure_output_dir() -> str:
    """
    Ensure part_b_skeleton/outputs/ exists. Returns absolute path.
    Safe to call on every run — mkdir -p semantics.
    """
    output_dir = os.path.join(_HERE, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def print_phase01_report(
    path_results: dict,
    const_violations: list,
    schema_violations: list,
    output_dir: str,
    elapsed_ms: float,
) -> None:
    """
    Print the Phase 01 scaffold validation report to stdout.
    Structured so an ISRO judge can read it in 10 seconds.
    """
    SEP = "═" * 60

    print(f"\n{SEP}")
    print("  PART B  ·  PHASE 01  ·  SCAFFOLD VALIDATION REPORT")
    print(SEP)

    # ── Section 1: Constants ──────────────────────────────────
    print("\n[1/4] LOCKED CONSTANTS")
    if not const_violations:
        print(f"  [OK]  TARGET_CRS          = {TARGET_CRS}")
        print(f"  [OK]  COLLAPSE_THRESHOLD  = {COLLAPSE_THRESHOLD}")
        print(f"  [OK]  TEST_TILE_BBOX      = {TEST_TILE_BBOX}")
        print(f"  [OK]  GRAPH_SOURCE        = '{GRAPH_SOURCE}'")
    else:
        for v in const_violations:
            print(f"  ✗  {v}")

    # ── Section 2: Schema ─────────────────────────────────────
    print("\n[2/4] CONTRACT SCHEMA (shared/schema.py)")
    if not schema_violations:
        print("  [OK]  RoadMaskMeta  — fields: crs, bbox, resolution_m, source")
        print("  [OK]  GraphNode     — fields: id, lat, lon")
        print("  [OK]  GraphEdge     — fields: source, target, weight_m, geometry")
        print("  [OK]  RoadGraph     — fields: nodes, edges, crs")
    else:
        for v in schema_violations:
            print(f"  ✗  {v}")

    # ── Section 3: Paths ──────────────────────────────────────
    print("\n[3/4] CONTRACT PATHS")
    for label, (abs_path, exists) in path_results.items():
        status = "[OK] EXISTS" if exists else "○ PENDING"
        # Show relative path for readability
        try:
            rel = os.path.relpath(abs_path, _REPO_ROOT)
        except ValueError:
            rel = abs_path
        print(f"  {status}  {rel}")
        print(f"           → {abs_path}")

    # ── Section 4: Output dir ─────────────────────────────────
    print("\n[4/4] OUTPUT DIRECTORY")
    print(f"  [OK]  {output_dir}")

    # ── Overall result ────────────────────────────────────────
    all_violations = const_violations + schema_violations
    print(f"\n{SEP}")
    if not all_violations:
        print(f"  RESULT: [OK] SCAFFOLD VALID  ({elapsed_ms:.1f} ms)")
        print(f"  CRS lock confirmed: {TARGET_CRS}")
        print(f"  Test tile: Koramangala, Bengaluru")
        print(f"  Ready for Phase 02 → contract validation in shared/eval.py")
    else:
        print(f"  RESULT: ✗ {len(all_violations)} VIOLATION(S) — fix before proceeding")
        for v in all_violations:
            print(f"    • {v}")
    print(SEP + "\n")


def main():
    t0 = time.perf_counter()
    timings_ms = {}   # Phase 25: per-stage benchmark timings

    print("Part B — Skeletonization & Topological Healing Engine")
    print("Mega-Heracross | ISRO Bharatiya Antariksh Hackathon 2026")
    print(f"Python {sys.version.split()[0]} | repo root: {_REPO_ROOT}")

    # Run all Phase 01 validations
    path_results      = validate_paths()
    const_violations  = validate_constants()
    schema_violations = validate_schema_imports()
    output_dir        = ensure_output_dir()

    elapsed_ms = (time.perf_counter() - t0) * 1000

    print_phase01_report(
        path_results, const_violations, schema_violations, output_dir, elapsed_ms
    )

    # ── Phase 02: contract validation ─────────────────────────────────────────
    # Path defined here, used after Phase 05 writes the file.
    graph_json_path = os.path.join(_REPO_ROOT, GRAPH_PATH)

    # ── Phase 17: source mode switch ──────────────────────────────────────────
    if is_osmnx_mode():
        # osmnx fallback — bypass mask/skeleton/sknw entirely
        road_graph, meta, osmnx_stats = run_osmnx_fallback()
        # Synthesise stub metrics for judge report compatibility
        mask = None
        skel_metrics   = {"skeleton_density": 0.0, "total_length_m": 0.0,
                          "n_components": 0}
        graph_stats    = {"n_nodes": len(road_graph.nodes),
                          "n_edges": len(road_graph.edges),
                          "total_length_km": osmnx_stats.get("total_length_km", 0),
                          "mean_weight_m": osmnx_stats.get("mean_weight_m", 0)}
        graph_violations = []
        loader_metrics = {"loader_pass": True}
        skel_violations = []
    else:
        # ── Phase 03: mask loader + geo-transform ─────────────────────────────
        mask_path = os.path.join(_REPO_ROOT, ROAD_MASK_PATH)
        meta_path = os.path.join(_REPO_ROOT, META_PATH)
        with StageTimer("loader", timings_ms):
            mask, meta, affine = load_inputs(
                mask_path, meta_path, use_synthetic_fallback=True
            )
            loader_metrics = print_loader_report(mask, meta, affine)

        # ── Phase 04: Zhang-Suen skeletonization ──────────────────────────────
        with StageTimer("skeletonize", timings_ms):
            skeleton, skel_metrics, skel_violations = run_skeletonization(
                mask, resolution_m=meta.resolution_m
            )

        # ── Phase 05: sknw graph extraction + RoadGraph emission ──────────────
        with StageTimer("graph_build", timings_ms):
            road_graph, graph_stats, graph_violations = build_and_save_graph(
                skeleton, affine, graph_json_path
            )

    # ── Phase 18: resolution-aware config ────────────────────────────────────
    res_cfg = make_config(meta.resolution_m, source=meta.source)
    print_resolution_config(res_cfg)

    # ── Phase 21: SAR-guided occlusion map ────────────────────────────────────
    # Only available in part_b mode (osmnx mode has no mask)
    if not is_osmnx_mode() and mask is not None:
        sar_mask_path = os.path.join(_REPO_ROOT, "part_a_vision/outputs/sar_mask.npy")
        occlusion_map, sar_metrics, sar_available = run_sar_integration(
            mask, affine, sar_mask_path=sar_mask_path
        )
    else:
        occlusion_map = None
        sar_available = False
        sar_metrics   = {}

    # ── Phases 10–12/20: topological healing ──────────────────────────────────
    # If SAR occlusion map available, use SAR-guided healing (Phase 21)
    # Otherwise fall back to standard spline healing (Phase 20)
    with StageTimer("healing", timings_ms):
        if occlusion_map is not None and occlusion_map.any():
            from part_b_skeleton.healer import detect_breaks
            break_pairs, detect_metrics = detect_breaks(
                road_graph, snap_m=res_cfg.snap_m
            )
            healed_graph, heal_metrics = sar_guided_heal(
                road_graph, break_pairs, occlusion_map, affine,
                snap_m=res_cfg.snap_m, lcc_target=0.80,
            )
            print_sar_report(sar_metrics, heal_metrics=heal_metrics,
                             sar_available=sar_available)
            # Run pruning after SAR-guided healing
            from part_b_skeleton.healer import prune_stubs, print_pruning_report
            healed_graph, prune_metrics = prune_stubs(
                healed_graph, resolution_m=res_cfg.effective_resolution_m
            )
            print_pruning_report(prune_metrics)
        else:
            healed_graph, heal_metrics = run_healing(
                road_graph,
                snap_m=res_cfg.snap_m,
                lcc_target=0.80,
                resolution_m=res_cfg.effective_resolution_m,
            )

    # Re-save healed graph as the final graph.json (replaces raw extraction)
    from part_b_skeleton.graph_builder import save_graph_json, compute_graph_stats
    save_graph_json(healed_graph, graph_json_path)
    graph_stats = compute_graph_stats(healed_graph)

    # ── Phase 14: degree-2 node collapse ──────────────────────────────────────
    with StageTimer("simplify", timings_ms):
        simplified_graph, simp_metrics = run_simplification(healed_graph)
        save_graph_json(simplified_graph, graph_json_path)
        graph_stats = compute_graph_stats(simplified_graph)

    # ── Phase 15: healing quality validation + LCC gate ───────────────────────
    val_result = healing_validation(
        simplified_graph,
        lcc_threshold=0.80,
        second_pass_snap_m=res_cfg.second_pass_snap_m,
        resolution_m=res_cfg.effective_resolution_m,
    )
    print_healing_validation(val_result)

    # If second pass ran and improved graph, use the healed version
    if val_result["passes_run"] == 2 and val_result["lcc_pass"]:
        final_graph = val_result["healed_graph"]
        save_graph_json(final_graph, graph_json_path)
        graph_stats = compute_graph_stats(final_graph)
    else:
        final_graph = simplified_graph

    # ── Phase 16: Haversine weight audit ─────────────────────────────────────
    with StageTimer("weight_audit", timings_ms):
        final_graph, weight_metrics = run_weight_audit(final_graph)
        save_graph_json(final_graph, graph_json_path)
        graph_stats = compute_graph_stats(final_graph)

    # ── Phase 22: Road type classification ───────────────────────────────────
    with StageTimer("classify", timings_ms):
        output_dir = os.path.join(_HERE, "outputs")
        road_types, class_metrics = run_classification(final_graph, output_dir)

    # ── Phase 23: Monte Carlo fragility score ────────────────────────────────
    with StageTimer("fragility", timings_ms):
        frag_result = graph_fragility_score(final_graph, n_removals=50)
        print_fragility_report(frag_result)

    # ── Phase 02: contract validation (re-run on freshly written graph.json) ──
    contract_result = validate_graph_contract(graph_json_path)
    print_contract_result(contract_result)

    # ── Phase 06: OSM ground truth download ───────────────────────────────────
    osm_graph, osm_stats = load_or_download_osm()

    # ── Phase 24: bootstrap confidence intervals ──────────────────────────────
    with StageTimer("bootstrap", timings_ms):
        boot_result = bootstrap_topology_ci(final_graph, osm_graph, n_boot=100)
        print_bootstrap_ci_report(boot_result)

    # ── Phase 07: topology F1 vs OSM ground truth ─────────────────────────────
    f1_result = graph_topology_f1(final_graph, osm_graph, snap_m=10.0)
    print_topology_f1_result(f1_result)

    # ── Phase 08: connected components analysis ───────────────────────────────
    conn_result = connectivity_report(final_graph)
    print_connectivity_report(conn_result)

    # ── Phase 09: judge-ready score report ────────────────────────────────────
    judge_metrics = {
        # Contract (Phase 02)
        "contract_status": contract_result["status"],
        "node_count":      contract_result["node_count"],
        "edge_count":      contract_result["edge_count"],
        # Loader (Phase 03)
        "source":          meta.source,
        "resolution_m":    meta.resolution_m,
        "mask_shape":      mask.shape if mask is not None else (0, 0),
        # Skeleton (Phase 04)
        "skeleton_density":    skel_metrics["skeleton_density"],
        "total_length_m":      skel_metrics["total_length_m"],
        "skeleton_components": skel_metrics["n_components"],
        # Graph (Phase 05 — healed)
        "n_nodes":         graph_stats["n_nodes"],
        "n_edges":         graph_stats["n_edges"],
        "total_length_km": graph_stats["total_length_km"],
        "mean_weight_m":   graph_stats["mean_weight_m"],
        # OSM (Phase 06)
        "osm_nodes":       osm_stats["n_nodes"],
        "osm_edges":       osm_stats["n_edges"],
        "osm_length_km":   osm_stats["total_length_km"],
        # Topology F1 (Phase 07)
        "node_f1":         f1_result["node_f1"],
        "edge_f1":         f1_result["edge_f1"],
        "snap_m":          f1_result["snap_m"],
        # Connectivity (Phase 08)
        "lcc_pct":         conn_result["lcc_pct"],
        "n_components":    conn_result["n_components"],
        "isolated_nodes":  conn_result["isolated_nodes"],
        "lcc_pass":        conn_result["lcc_pass"],
        # Healing (Phases 10-12)
        "healed_edges":    heal_metrics.get("healed_edges", 0),
        "lcc_before":      heal_metrics.get("lcc_before", 0),
        "lcc_after":       heal_metrics.get("lcc_after", 0),
        # Road types (Phase 22)
        "pct_highway":     class_metrics.get("pct_highway", 0),
        "pct_arterial":    class_metrics.get("pct_arterial", 0),
        "pct_local":       class_metrics.get("pct_local", 0),
    }
    print_judge_report(judge_metrics)

    # ── Phase 19: end-to-end integration test ────────────────────────────────
    integration_passed = run_integration_test(verbose=False)

    # ── Phase 25: full pipeline benchmark + reproducibility report ───────────
    total_runtime_s = time.perf_counter() - t0

    benchmark_metrics = dict(judge_metrics)
    benchmark_metrics.update({
        "fragility_auc":  frag_result.get("fragility_auc", 0),
        "node_f1_ci":     list(boot_result.get("node_f1_ci", (0, 0))),
        "edge_f1_ci":     list(boot_result.get("edge_f1_ci", (0, 0))),
        "timings_ms":     timings_ms,
        "total_runtime_s": round(total_runtime_s, 2),
    })

    mask_path_for_hash = os.path.join(_REPO_ROOT, ROAD_MASK_PATH)
    meta_path_for_hash = os.path.join(_REPO_ROOT, META_PATH)
    benchmark_output_path = os.path.join(_HERE, "outputs", "part_b_benchmark.json")

    benchmark = run_benchmark(
        mask_path_for_hash, meta_path_for_hash,
        graph_json_path, benchmark_metrics,
        benchmark_output_path,
    )
    print_benchmark_report(benchmark)

    # Exit with non-zero status if any violation found
    # so CI/CD pipelines can catch scaffold failures
    n_violations = len(const_violations) + len(schema_violations)
    loader_ok    = loader_metrics.get("loader_pass", True)
    skeleton_ok  = len(skel_violations) == 0
    graph_ok     = len(graph_violations) == 0
    contract_ok  = contract_result["status"] == "PASS"
    # Note: connectivity failure is reported but does NOT block exit —
    # healing (Phase 12) is expected to fix LCC% before Part C runs
    sys.exit(0 if (n_violations == 0 and loader_ok and skeleton_ok
                   and graph_ok and contract_ok
                   and integration_passed) else 1)


if __name__ == "__main__":
    main()

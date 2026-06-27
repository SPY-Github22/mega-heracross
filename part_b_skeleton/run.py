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
    Phase 01 ✓  Repo scaffold & contract wiring
    Phase 02 ✓  Contract validation in shared/eval.py
    Phase 03 ✓  Synthetic mask loader + geo-transform
    Phase 04 ✓  Zhang-Suen skeletonization
    Phase 05 ✓  sknw graph extraction + RoadGraph emission
    Phase 06 ✓  OSM ground truth download for Koramangala
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
from shared.eval import validate_graph_contract, print_contract_result
from part_b_skeleton.loader import load_inputs, print_loader_report
from part_b_skeleton.skeletonize import run_skeletonization
from part_b_skeleton.graph_builder import build_and_save_graph
from part_b_skeleton.osm_reference import load_or_download_osm
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
        print(f"  ✓  TARGET_CRS          = {TARGET_CRS}")
        print(f"  ✓  COLLAPSE_THRESHOLD  = {COLLAPSE_THRESHOLD}")
        print(f"  ✓  TEST_TILE_BBOX      = {TEST_TILE_BBOX}")
        print(f"  ✓  GRAPH_SOURCE        = '{GRAPH_SOURCE}'")
    else:
        for v in const_violations:
            print(f"  ✗  {v}")

    # ── Section 2: Schema ─────────────────────────────────────
    print("\n[2/4] CONTRACT SCHEMA (shared/schema.py)")
    if not schema_violations:
        print("  ✓  RoadMaskMeta  — fields: crs, bbox, resolution_m, source")
        print("  ✓  GraphNode     — fields: id, lat, lon")
        print("  ✓  GraphEdge     — fields: source, target, weight_m, geometry")
        print("  ✓  RoadGraph     — fields: nodes, edges, crs")
    else:
        for v in schema_violations:
            print(f"  ✗  {v}")

    # ── Section 3: Paths ──────────────────────────────────────
    print("\n[3/4] CONTRACT PATHS")
    for label, (abs_path, exists) in path_results.items():
        status = "✓ EXISTS" if exists else "○ PENDING"
        # Show relative path for readability
        try:
            rel = os.path.relpath(abs_path, _REPO_ROOT)
        except ValueError:
            rel = abs_path
        print(f"  {status}  {rel}")
        print(f"           → {abs_path}")

    # ── Section 4: Output dir ─────────────────────────────────
    print("\n[4/4] OUTPUT DIRECTORY")
    print(f"  ✓  {output_dir}")

    # ── Overall result ────────────────────────────────────────
    all_violations = const_violations + schema_violations
    print(f"\n{SEP}")
    if not all_violations:
        print(f"  RESULT: ✓ SCAFFOLD VALID  ({elapsed_ms:.1f} ms)")
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

    # ── Phase 03: mask loader + geo-transform ──────────────────────────────────
    mask_path = os.path.join(_REPO_ROOT, ROAD_MASK_PATH)
    meta_path = os.path.join(_REPO_ROOT, META_PATH)
    mask, meta, affine = load_inputs(
        mask_path, meta_path, use_synthetic_fallback=True
    )
    loader_metrics = print_loader_report(mask, meta, affine)

    # ── Phase 04: Zhang-Suen skeletonization ──────────────────────────────────
    skeleton, skel_metrics, skel_violations = run_skeletonization(
        mask, resolution_m=meta.resolution_m
    )

    # ── Phase 05: sknw graph extraction + RoadGraph emission ──────────────────
    graph_json_path = os.path.join(_REPO_ROOT, GRAPH_PATH)
    road_graph, graph_stats, graph_violations = build_and_save_graph(
        skeleton, affine, graph_json_path
    )

    # ── Phase 02: contract validation (re-run on freshly written graph.json) ──
    contract_result = validate_graph_contract(graph_json_path)
    print_contract_result(contract_result)

    # ── Phase 06: OSM ground truth download ───────────────────────────────────
    osm_graph, osm_stats = load_or_download_osm()

    # Exit with non-zero status if any violation found
    # so CI/CD pipelines can catch scaffold failures
    n_violations = len(const_violations) + len(schema_violations)
    loader_ok   = loader_metrics.get("loader_pass", True)
    skeleton_ok = len(skel_violations) == 0
    graph_ok    = len(graph_violations) == 0
    contract_ok = contract_result["status"] == "PASS"
    sys.exit(0 if (n_violations == 0 and loader_ok and skeleton_ok
                   and graph_ok and contract_ok) else 1)


if __name__ == "__main__":
    main()

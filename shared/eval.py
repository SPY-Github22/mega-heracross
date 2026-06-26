"""
shared/eval.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Shared evaluation module — all three Parts contribute here.
Runs automatically at the end of every Part B execution.

EVALUATION LAYERS:
    Layer 1 — Contract validation        (Phase 02) ✓
    Layer 2 — Quantitative topology      (Phases 07–09)
    Layer 3 — Statistical confidence     (Phases 23–25)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
import os
from typing import Any


# ══════════════════════════════════════════════════════════════
# LAYER 1 — CONTRACT VALIDATION  (Phase 02)
# ══════════════════════════════════════════════════════════════

# Koramangala bounding box for lat/lon range checks
# (min_lon, min_lat, max_lon, max_lat) with 0.05° buffer
_BBOX_BUFFER = 0.05
_LON_MIN = 77.6101 - _BBOX_BUFFER
_LON_MAX = 77.6401 + _BBOX_BUFFER
_LAT_MIN = 12.9177 - _BBOX_BUFFER
_LAT_MAX = 12.9377 + _BBOX_BUFFER


def _err(violations: list, msg: str) -> None:
    """Append a violation message."""
    violations.append(msg)


def _check_top_level(data: Any, violations: list) -> bool:
    """
    Check that graph.json is a dict with exactly the right top-level keys.
    Returns False if structure is so broken that further checks can't run.
    """
    if not isinstance(data, dict):
        _err(violations, f"graph.json must be a JSON object (dict), got {type(data).__name__}")
        return False

    required_keys = {"nodes", "edges", "crs"}
    present_keys  = set(data.keys())
    missing = required_keys - present_keys
    extra   = present_keys - required_keys

    for k in sorted(missing):
        _err(violations, f"Missing required top-level key: '{k}'")
    for k in sorted(extra):
        _err(violations, f"Unexpected top-level key: '{k}' (schema has no such field)")

    return len(missing) == 0   # can only continue if required keys exist


def _check_crs(data: dict, violations: list) -> None:
    """CRS must be exactly 'EPSG:4326'. No exceptions."""
    crs = data.get("crs")
    if crs is None:
        return  # already caught by _check_top_level
    if not isinstance(crs, str):
        _err(violations, f"crs must be a string, got {type(crs).__name__}: {crs!r}")
    elif crs != "EPSG:4326":
        _err(violations, f"crs must be 'EPSG:4326', got '{crs}' — CRS is a locked constant")


def _check_nodes(data: dict, violations: list) -> set:
    """
    Validate every node in the nodes list.
    Returns the set of valid node IDs (used later for edge cross-reference).
    """
    nodes = data.get("nodes", [])
    valid_ids = set()

    if not isinstance(nodes, list):
        _err(violations, f"'nodes' must be a list, got {type(nodes).__name__}")
        return valid_ids

    if len(nodes) == 0:
        _err(violations, "'nodes' list is empty — a road graph must have nodes")
        return valid_ids

    required_node_keys = {"id", "lat", "lon"}

    for i, node in enumerate(nodes):
        prefix = f"nodes[{i}]"

        if not isinstance(node, dict):
            _err(violations, f"{prefix}: must be a dict, got {type(node).__name__}")
            continue

        # Check required fields exist
        missing = required_node_keys - set(node.keys())
        for k in sorted(missing):
            _err(violations, f"{prefix}: missing required field '{k}'")
        if missing:
            continue

        # id: must be int
        node_id = node["id"]
        if not isinstance(node_id, int):
            _err(violations, f"{prefix}: 'id' must be int, got {type(node_id).__name__}: {node_id!r}")
        else:
            valid_ids.add(node_id)

        # lat: must be float/int, within plausible range for Koramangala
        lat = node["lat"]
        if not isinstance(lat, (int, float)):
            _err(violations, f"{prefix}: 'lat' must be numeric, got {type(lat).__name__}: {lat!r}")
        elif not (_LAT_MIN <= lat <= _LAT_MAX):
            _err(violations,
                 f"{prefix}: lat={lat} is outside expected range "
                 f"[{_LAT_MIN:.4f}, {_LAT_MAX:.4f}] for Koramangala — "
                 f"check geo-transform (EPSG:4326 expected, not pixel coords)")

        # lon: must be float/int, within plausible range for Koramangala
        lon = node["lon"]
        if not isinstance(lon, (int, float)):
            _err(violations, f"{prefix}: 'lon' must be numeric, got {type(lon).__name__}: {lon!r}")
        elif not (_LON_MIN <= lon <= _LON_MAX):
            _err(violations,
                 f"{prefix}: lon={lon} is outside expected range "
                 f"[{_LON_MIN:.4f}, {_LON_MAX:.4f}] for Koramangala — "
                 f"check geo-transform (EPSG:4326 expected, not pixel coords)")

    return valid_ids


def _check_edges(data: dict, valid_node_ids: set, violations: list) -> None:
    """
    Validate every edge in the edges list.
    Also checks that source/target reference real node IDs.
    """
    edges = data.get("edges", [])

    if not isinstance(edges, list):
        _err(violations, f"'edges' must be a list, got {type(edges).__name__}")
        return

    if len(edges) == 0:
        _err(violations, "'edges' list is empty — a road graph must have edges")
        return

    required_edge_keys = {"source", "target", "weight_m", "geometry"}

    for i, edge in enumerate(edges):
        prefix = f"edges[{i}]"

        if not isinstance(edge, dict):
            _err(violations, f"{prefix}: must be a dict, got {type(edge).__name__}")
            continue

        missing = required_edge_keys - set(edge.keys())
        for k in sorted(missing):
            _err(violations, f"{prefix}: missing required field '{k}'")
        if missing:
            continue

        # source / target: must be int and must reference a known node
        for key in ("source", "target"):
            val = edge[key]
            if not isinstance(val, int):
                _err(violations, f"{prefix}: '{key}' must be int, got {type(val).__name__}: {val!r}")
            elif valid_node_ids and val not in valid_node_ids:
                _err(violations,
                     f"{prefix}: {key}={val} references a node ID that does not exist in 'nodes'")

        # weight_m: must be positive float/int
        w = edge["weight_m"]
        if not isinstance(w, (int, float)):
            _err(violations, f"{prefix}: 'weight_m' must be numeric, got {type(w).__name__}: {w!r}")
        elif w <= 0:
            _err(violations, f"{prefix}: 'weight_m' must be > 0, got {w} — "
                             f"a road segment with zero or negative length is invalid")

        # geometry: must be a non-empty list of [lat, lon] pairs
        geom = edge["geometry"]
        if not isinstance(geom, list):
            _err(violations, f"{prefix}: 'geometry' must be a list, got {type(geom).__name__}")
        elif len(geom) == 0:
            _err(violations, f"{prefix}: 'geometry' is empty — "
                             f"every edge needs at least its two endpoint coordinates")
        else:
            for j, pt in enumerate(geom):
                pt_prefix = f"{prefix}.geometry[{j}]"
                if not isinstance(pt, (list, tuple)) or len(pt) != 2:
                    _err(violations,
                         f"{pt_prefix}: each geometry point must be [lat, lon] (length-2 list), "
                         f"got {pt!r}")
                    break  # one bad point per edge is enough to report
                lat_p, lon_p = pt
                if not isinstance(lat_p, (int, float)) or not isinstance(lon_p, (int, float)):
                    _err(violations,
                         f"{pt_prefix}: lat/lon must be numeric, got [{type(lat_p).__name__}, "
                         f"{type(lon_p).__name__}]")


def validate_graph_contract(graph_json_path: str) -> dict:
    """
    Layer 1: Contract validation — runs on EVERY execution.

    Loads graph.json and checks it matches shared/schema.RoadGraph exactly:
      • Top-level keys: nodes, edges, crs
      • crs == 'EPSG:4326'
      • Every node has id (int), lat (float), lon (float) in Koramangala range
      • Every edge has source (int), target (int), weight_m (float > 0),
        geometry (non-empty list of [lat, lon] pairs)
      • Every edge source/target references a real node id

    Returns
    -------
    dict with keys:
        status        : "PASS" | "FAIL" | "FILE_NOT_FOUND" | "JSON_ERROR"
        violations    : list of violation strings (empty on PASS)
        node_count    : int  (0 if file unreadable)
        edge_count    : int  (0 if file unreadable)
        summary       : human-readable one-line result string
    """
    result = {
        "status":     "FAIL",
        "violations": [],
        "node_count": 0,
        "edge_count": 0,
        "summary":    "",
    }
    violations = result["violations"]

    # ── 1. File existence ────────────────────────────────────
    if not os.path.exists(graph_json_path):
        result["status"]  = "FILE_NOT_FOUND"
        result["summary"] = f"CONTRACT: FILE_NOT_FOUND — {graph_json_path}"
        return result

    # ── 2. JSON parse ────────────────────────────────────────
    try:
        with open(graph_json_path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        result["status"]  = "JSON_ERROR"
        result["summary"] = f"CONTRACT: JSON_ERROR — {e}"
        violations.append(str(e))
        return result

    # ── 3. Structural checks ──────────────────────────────────
    can_continue = _check_top_level(data, violations)

    if can_continue:
        _check_crs(data, violations)
        valid_node_ids = _check_nodes(data, violations)
        _check_edges(data, valid_node_ids, violations)

        # Populate counts even if there are violations (useful for diagnostics)
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])
        if isinstance(nodes, list):
            result["node_count"] = len(nodes)
        if isinstance(edges, list):
            result["edge_count"] = len(edges)

    # ── 4. Final verdict ──────────────────────────────────────
    if violations:
        result["status"]  = "FAIL"
        result["summary"] = (
            f"CONTRACT: FAIL — {len(violations)} violation(s) | "
            f"nodes={result['node_count']} edges={result['edge_count']}"
        )
    else:
        result["status"]  = "PASS"
        result["summary"] = (
            f"CONTRACT: PASS | "
            f"nodes={result['node_count']} edges={result['edge_count']} | "
            f"CRS=EPSG:4326 ✓"
        )

    return result


def print_contract_result(result: dict) -> None:
    """
    Print the contract validation result in a format an ISRO judge can read
    in under 5 seconds. Called automatically at end of every run.
    """
    SEP = "─" * 60
    print(f"\n{SEP}")
    print(f"  EVAL LAYER 1 — CONTRACT VALIDATION")
    print(SEP)
    print(f"  {result['summary']}")

    if result["violations"]:
        print(f"\n  Violations ({len(result['violations'])}):")
        for v in result["violations"]:
            print(f"    ✗ {v}")
    else:
        status = result["status"]
        if status == "PASS":
            print(f"  All schema fields present and correctly typed.")
            print(f"  Node lat/lon within Koramangala bounding box ✓")
            print(f"  All edge source/target IDs reference valid nodes ✓")
            print(f"  All edge weight_m > 0 ✓")
            print(f"  All geometry lists non-empty ✓")

    print(SEP)


# ══════════════════════════════════════════════════════════════
# LAYER 2 — QUANTITATIVE TOPOLOGY  (Phases 07–09, stubs)
# ══════════════════════════════════════════════════════════════

def graph_topology_f1(our_graph, osm_graph, snap_m: float = 10.0) -> dict:
    """Phase 07: Node/edge F1 vs OSM ground truth."""
    return {"status": "NOT_IMPLEMENTED", "phase": "07"}


def connectivity_report(graph) -> dict:
    """Phase 08: Connected components analysis."""
    return {"status": "NOT_IMPLEMENTED", "phase": "08"}


def print_judge_report(metrics: dict) -> None:
    """Phase 09: 10-second ISRO judge summary."""
    print("\n[eval.py] Full judge report arrives Phase 09.")


# ══════════════════════════════════════════════════════════════
# LAYER 3 — STATISTICAL CONFIDENCE  (Phases 23–25, stubs)
# ══════════════════════════════════════════════════════════════

def graph_fragility_score(graph, n_removals: int = 50) -> dict:
    """Phase 23: Monte Carlo edge-removal fragility curve."""
    return {"status": "NOT_IMPLEMENTED", "phase": "23"}


def bootstrap_topology_ci(our_graph, osm_graph, n_boot: int = 100) -> dict:
    """Phase 24: Bootstrap confidence intervals on topology metrics."""
    return {"status": "NOT_IMPLEMENTED", "phase": "24"}

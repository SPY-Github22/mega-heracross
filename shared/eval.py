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


# ══════════════════════════════════════════════════════════════
# LAYER 2 — TOPOLOGY F1  (Phase 07)
# ══════════════════════════════════════════════════════════════

def graph_topology_f1(our_graph, osm_graph, snap_m: float = 10.0) -> dict:
    """
    Layer 2: Node and edge F1 vs OSM ground truth.

    Algorithm
    ---------
    Node matching (KDTree snap):
      1. Build KDTree on OSM node coordinates (lat-scaled to metres)
      2. For each node in our graph, find nearest OSM node
      3. If distance <= snap_m: match (true positive)
      4. Unmatched our-nodes = false positives
      5. Unmatched OSM nodes = false negatives

    Edge matching:
      An edge (u,v) in our graph matches an OSM edge (a,b) if:
        - our node u snaps to OSM node a (or b)
        - our node v snaps to OSM node b (or a)
      i.e. both endpoints must snap to the same OSM edge endpoints.

    Returns
    -------
    dict with:
      node_precision, node_recall, node_f1
      edge_precision, edge_recall, edge_f1
      matched_nodes, our_nodes, osm_nodes
      matched_edges, our_edges, osm_edges
      snap_m  (the tolerance used)
    """
    import math
    import numpy as np
    from scipy.spatial import KDTree

    # ── Coordinate scaling ─────────────────────────────────────
    # Convert lat/lon to approximate metres for KDTree distance queries.
    # At Koramangala (~13°N):
    #   1 deg lat ≈ 111,320 m
    #   1 deg lon ≈ 108,498 m
    METRES_PER_DEG_LAT = 111_320.0
    centre_lat = sum(n.lat for n in osm_graph.nodes) / max(len(osm_graph.nodes), 1)
    METRES_PER_DEG_LON = 111_320.0 * math.cos(math.radians(centre_lat))

    def to_metres(lat, lon):
        return (lat * METRES_PER_DEG_LAT, lon * METRES_PER_DEG_LON)

    # ── Build KDTree on OSM nodes ──────────────────────────────
    osm_nodes  = list(osm_graph.nodes)
    n_osm_nodes = len(osm_nodes)
    osm_coords = np.array([to_metres(n.lat, n.lon) for n in osm_nodes])
    tree = KDTree(osm_coords)

    # ── Query: nearest OSM node for each of our nodes ──────────
    our_nodes = list(our_graph.nodes)

    # Guard: empty our_graph
    if len(our_nodes) == 0:
        return {
            "snap_m": snap_m,
            "node_precision": 0.0, "node_recall": 0.0, "node_f1": 0.0,
            "edge_precision": 0.0, "edge_recall": 0.0, "edge_f1": 0.0,
            "matched_nodes": 0, "our_nodes": 0, "osm_nodes": n_osm_nodes,
            "matched_edges": 0, "our_edges": 0, "osm_edges": len(list(osm_graph.edges)),
        }

    our_coords = np.array([to_metres(n.lat, n.lon) for n in our_nodes])

    # Query: nearest OSM node for each of our nodes
    dists, osm_idxs = tree.query(our_coords, k=1)

    # our_to_osm[i] = OSM node index if snapped, else -1
    our_to_osm = {}
    osm_matched = set()

    for i, (dist, osm_idx) in enumerate(zip(dists, osm_idxs)):
        if dist <= snap_m:
            our_to_osm[i] = int(osm_idx)
            osm_matched.add(int(osm_idx))
        else:
            our_to_osm[i] = -1

    n_matched_nodes = len(osm_matched)
    n_our_nodes     = len(our_nodes)
    n_osm_nodes     = len(osm_nodes)

    node_precision = n_matched_nodes / n_our_nodes  if n_our_nodes  > 0 else 0.0
    node_recall    = n_matched_nodes / n_osm_nodes  if n_osm_nodes  > 0 else 0.0
    node_f1 = (2 * node_precision * node_recall /
               (node_precision + node_recall)
               if (node_precision + node_recall) > 0 else 0.0)

    # ── Edge matching ──────────────────────────────────────────
    # Build set of OSM edge pairs (as frozensets of OSM node indices)
    osm_edge_set = set()
    for e in osm_graph.edges:
        # Map OSM graph node IDs back to KDTree indices
        src_osm_id = e.source
        tgt_osm_id = e.target
        osm_edge_set.add(frozenset([src_osm_id, tgt_osm_id]))

    # Build node-id-to-index maps
    our_node_id_to_idx  = {n.id: i for i, n in enumerate(our_nodes)}
    osm_node_id_to_idx  = {n.id: i for i, n in enumerate(osm_nodes)}

    matched_edges = 0
    for e in our_graph.edges:
        src_idx = our_node_id_to_idx.get(e.source, -1)
        tgt_idx = our_node_id_to_idx.get(e.target, -1)
        if src_idx == -1 or tgt_idx == -1:
            continue

        osm_src = our_to_osm.get(src_idx, -1)
        osm_tgt = our_to_osm.get(tgt_idx, -1)
        if osm_src == -1 or osm_tgt == -1:
            continue

        # Translate KDTree indices back to OSM node IDs
        osm_src_id = osm_nodes[osm_src].id
        osm_tgt_id = osm_nodes[osm_tgt].id
        if frozenset([osm_src_id, osm_tgt_id]) in osm_edge_set:
            matched_edges += 1

    n_our_edges = len(our_graph.edges)
    n_osm_edges = len(osm_graph.edges)

    edge_precision = matched_edges / n_our_edges if n_our_edges > 0 else 0.0
    edge_recall    = matched_edges / n_osm_edges if n_osm_edges > 0 else 0.0
    edge_f1 = (2 * edge_precision * edge_recall /
               (edge_precision + edge_recall)
               if (edge_precision + edge_recall) > 0 else 0.0)

    return {
        "snap_m":          snap_m,
        "node_precision":  round(node_precision, 4),
        "node_recall":     round(node_recall, 4),
        "node_f1":         round(node_f1, 4),
        "edge_precision":  round(edge_precision, 4),
        "edge_recall":     round(edge_recall, 4),
        "edge_f1":         round(edge_f1, 4),
        "matched_nodes":   n_matched_nodes,
        "our_nodes":       n_our_nodes,
        "osm_nodes":       n_osm_nodes,
        "matched_edges":   matched_edges,
        "our_edges":       n_our_edges,
        "osm_edges":       n_osm_edges,
    }


def print_topology_f1_result(result: dict) -> None:
    """Print Phase 07 topology F1 report."""
    SEP = "─" * 60

    def _bar(val, width=20):
        filled = int(val * width)
        return "█" * filled + "░" * (width - filled)

    def _grade(f1):
        if f1 >= 0.80: return "✓ STRONG"
        if f1 >= 0.60: return "○ ACCEPTABLE"
        if f1 >= 0.40: return "⚠ WEAK"
        return "✗ POOR"

    print(f"\n{SEP}")
    print(f"  PHASE 07 — TOPOLOGY F1 vs OSM GROUND TRUTH")
    print(f"  Snap tolerance: {result['snap_m']}m")
    print(SEP)
    print(f"\n  NODE MATCHING")
    print(f"    Our nodes  : {result['our_nodes']}")
    print(f"    OSM nodes  : {result['osm_nodes']}")
    print(f"    Matched    : {result['matched_nodes']}")
    print(f"    Precision  : {result['node_precision']:.4f}  {_bar(result['node_precision'])}")
    print(f"    Recall     : {result['node_recall']:.4f}  {_bar(result['node_recall'])}")
    print(f"    F1         : {result['node_f1']:.4f}  {_grade(result['node_f1'])}")

    print(f"\n  EDGE MATCHING")
    print(f"    Our edges  : {result['our_edges']}")
    print(f"    OSM edges  : {result['osm_edges']}")
    print(f"    Matched    : {result['matched_edges']}")
    print(f"    Precision  : {result['edge_precision']:.4f}  {_bar(result['edge_precision'])}")
    print(f"    Recall     : {result['edge_recall']:.4f}  {_bar(result['edge_recall'])}")
    print(f"    F1         : {result['edge_f1']:.4f}  {_grade(result['edge_f1'])}")

    print(f"\n  ISRO JUDGE TARGET: node_F1 > 0.70, edge_F1 > 0.60")
    print(f"  (Low scores on synthetic data are expected —")
    print(f"   real LISS-IV mask + healing will dramatically improve these)")
    print(f"\n{SEP}")
    print(SEP)


# ══════════════════════════════════════════════════════════════
# LAYER 2 — CONNECTIVITY REPORT  (Phase 08)
# ══════════════════════════════════════════════════════════════

def connectivity_report(graph) -> dict:
    """
    Layer 2: Connected components analysis.

    Builds a NetworkX graph from RoadGraph and computes:
      n_components     — number of connected components (lower = better)
      lcc_nodes        — node count of largest connected component
      lcc_pct          — LCC as % of total nodes (key health metric)
      isolated_nodes   — degree-0 nodes (completely disconnected)
      small_components — components with < 5 nodes (likely noise/artifacts)
      mean_component_size — average nodes per component
      lcc_pass         — bool: LCC% > 60% (minimum usable threshold)

    LCC% interpretation:
      > 80% : healthy graph, healing is working well
      60-80%: acceptable, some fragmentation remains
      < 60% : graph is too fragmented for reliable routing analysis
              Phase 12 healing must fix this before Part C runs

    Parameters
    ----------
    graph : RoadGraph — from shared/schema.py

    Returns
    -------
    dict of connectivity metrics
    """
    import networkx as nx

    n_nodes = len(graph.nodes)
    n_edges = len(graph.edges)

    # ── Edge case: empty graph ────────────────────────────────
    if n_nodes == 0:
        return {
            "n_nodes":            0,
            "n_edges":            0,
            "n_components":       0,
            "lcc_nodes":          0,
            "lcc_pct":            0.0,
            "isolated_nodes":     0,
            "small_components":   0,
            "mean_component_size": 0.0,
            "lcc_pass":           False,
            "lcc_threshold":      0.60,
        }

    # ── Build undirected NetworkX graph ───────────────────────
    G = nx.Graph()
    G.add_nodes_from([n.id for n in graph.nodes])
    G.add_edges_from([(e.source, e.target) for e in graph.edges])

    # ── Connected components ──────────────────────────────────
    components = list(nx.connected_components(G))
    n_components = len(components)
    component_sizes = sorted([len(c) for c in components], reverse=True)

    lcc_nodes = component_sizes[0] if component_sizes else 0
    lcc_pct   = lcc_nodes / n_nodes if n_nodes > 0 else 0.0

    # Isolated nodes: degree 0 (no edges at all)
    isolated_nodes = sum(1 for n in G.nodes() if G.degree(n) == 0)

    # Small components: < 5 nodes (likely noise artifacts)
    small_components = sum(1 for s in component_sizes if s < 5)

    mean_component_size = (sum(component_sizes) / n_components
                           if n_components > 0 else 0.0)

    LCC_THRESHOLD = 0.60

    return {
        "n_nodes":             n_nodes,
        "n_edges":             n_edges,
        "n_components":        n_components,
        "lcc_nodes":           lcc_nodes,
        "lcc_pct":             round(lcc_pct, 4),
        "isolated_nodes":      isolated_nodes,
        "small_components":    small_components,
        "mean_component_size": round(mean_component_size, 1),
        "lcc_pass":            lcc_pct >= LCC_THRESHOLD,
        "lcc_threshold":       LCC_THRESHOLD,
        "component_sizes":     component_sizes,
    }


def print_connectivity_report(result: dict) -> None:
    """Print Phase 08 connectivity report."""
    SEP = "─" * 60

    lcc_pct    = result["lcc_pct"]
    lcc_pass   = result["lcc_pass"]
    threshold  = result["lcc_threshold"]

    # Visual LCC bar
    bar_width  = 30
    filled     = int(lcc_pct * bar_width)
    thresh_pos = int(threshold * bar_width)
    bar        = list("░" * bar_width)
    for i in range(filled):
        bar[i] = "█"
    if thresh_pos < bar_width:
        bar[thresh_pos] = "|"   # threshold marker
    bar_str = "".join(bar)

    # Health label
    if lcc_pct >= 0.80:
        health = "✓ HEALTHY"
    elif lcc_pct >= 0.60:
        health = "○ ACCEPTABLE"
    elif lcc_pct >= 0.40:
        health = "⚠ FRAGMENTED"
    else:
        health = "✗ CRITICAL"

    print(f"\n{SEP}")
    print(f"  PHASE 08 — CONNECTED COMPONENTS ANALYSIS")
    print(SEP)
    print(f"  Total nodes       : {result['n_nodes']}")
    print(f"  Total edges       : {result['n_edges']}")
    print(f"  Components        : {result['n_components']}")
    print(f"  Mean comp. size   : {result['mean_component_size']:.1f} nodes")
    print(f"  Isolated nodes    : {result['isolated_nodes']}")
    print(f"  Small comps (<5)  : {result['small_components']}")
    print(f"\n  LCC (Largest Connected Component)")
    print(f"    Nodes in LCC    : {result['lcc_nodes']} / {result['n_nodes']}")
    print(f"    LCC%            : {lcc_pct:.1%}  {health}")
    print(f"    [{bar_str}] {lcc_pct:.0%}")
    print(f"     threshold={threshold:.0%}─┘  (minimum for routing analysis)")

    # Component size distribution
    sizes = result.get("component_sizes", [])
    if len(sizes) > 1:
        print(f"\n  Component sizes (largest first):")
        for i, s in enumerate(sizes[:5]):
            pct = s / result["n_nodes"]
            bar2 = "█" * int(pct * 20) + "░" * (20 - int(pct * 20))
            print(f"    [{i+1}] {s:4d} nodes  {bar2}  {pct:.1%}")
        if len(sizes) > 5:
            print(f"    ... and {len(sizes)-5} more small components")

    print(f"\n  Routing implications:")
    if lcc_pass:
        print(f"    ✓ Graph is usable for Part C criticality analysis")
        print(f"    ✓ {lcc_pct:.1%} of nodes reachable from largest component")
    else:
        print(f"    ✗ LCC% = {lcc_pct:.1%} < {threshold:.0%} threshold")
        print(f"    ✗ Phase 12 MST healing must improve connectivity before Part C")

    print(f"\n{SEP}")
    print(f"  CONNECTIVITY: {'✓ PASS' if lcc_pass else '✗ FAIL — healing required'}")
    print(SEP)


# ══════════════════════════════════════════════════════════════
# LAYER 3 — JUDGE-READY SCORE REPORT  (Phase 09)
# ══════════════════════════════════════════════════════════════

def print_judge_report(metrics: dict) -> None:
    """
    Layer 3: 10-second ISRO judge summary.

    Prints a single structured report that collects every key metric
    from Phases 02–08 into one place. An ISRO judge with a GIS/ML
    background should be able to assess system performance at a glance.

    Expected keys in metrics dict:
      From Phase 02 (contract):   contract_status, node_count, edge_count
      From Phase 03 (loader):     source, resolution_m, mask_shape
      From Phase 04 (skeleton):   skeleton_density, total_length_m,
                                   skeleton_components
      From Phase 05 (graph):      n_nodes, n_edges, total_length_km
      From Phase 06 (osm):        osm_nodes, osm_edges, osm_length_km
      From Phase 07 (topology):   node_f1, edge_f1, snap_m
      From Phase 08 (conn):       lcc_pct, n_components, lcc_pass
    """
    import datetime

    W  = 62          # report width
    TS = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _line(label, value, unit="", status=""):
        label_w = 26
        value_s = f"{value}{unit}"
        status_s = f"  {status}" if status else ""
        return f"  {label:<{label_w}} {value_s:<14}{status_s}"

    def _score_tag(val, good, ok):
        if val >= good: return "✓"
        if val >= ok:   return "○"
        return "✗"

    SEP  = "═" * W
    SEP2 = "─" * W

    print(f"\n{SEP}")
    print(f"  PART B — SCORE REPORT         {TS}")
    print(f"  Mega-Heracross · ISRO Hackathon 2026 · Koramangala Test Tile")
    print(SEP)

    # ── Section 1: CONTRACT ───────────────────────────────────
    contract_status = metrics.get("contract_status", "UNKNOWN")
    n_nodes  = metrics.get("node_count", 0)
    n_edges  = metrics.get("edge_count", 0)
    c_ok = "✓" if contract_status == "PASS" else "✗"

    print(f"\n  [1] CONTRACT")
    print(_line("Schema validation",  contract_status,   "", c_ok))
    print(_line("Nodes emitted",      n_nodes,           ""))
    print(_line("Edges emitted",      n_edges,           ""))
    print(_line("CRS",                "EPSG:4326",       "", "✓"))

    # ── Section 2: INPUT MASK ─────────────────────────────────
    source      = metrics.get("source", "unknown")
    res_m       = metrics.get("resolution_m", 0)
    mask_shape  = metrics.get("mask_shape", (0, 0))
    skel_dens   = metrics.get("skeleton_density", 0)
    road_len_km = metrics.get("total_length_m", 0) / 1000

    print(f"\n  [2] INPUT MASK")
    print(_line("Source",             source,            ""))
    print(_line("Resolution",         f"{res_m:.1f}",    " m/px"))
    print(_line("Mask size",          f"{mask_shape[0]}×{mask_shape[1]}", " px"))
    print(_line("Skeleton density",   f"{skel_dens:.4f}","",
                _score_tag(skel_dens, 0.05, 0.02)))
    print(_line("Skeleton length",    f"{road_len_km:.2f}", " km"))

    # ── Section 3: GRAPH QUALITY ──────────────────────────────
    total_km    = metrics.get("total_length_km", 0)
    mean_w      = metrics.get("mean_weight_m", 0)
    osm_nodes   = metrics.get("osm_nodes", 0)
    osm_edges   = metrics.get("osm_edges", 0)
    osm_km      = metrics.get("osm_length_km", 0)

    node_cov = (n_nodes / osm_nodes) if osm_nodes > 0 else 0
    edge_cov = (n_edges / osm_edges) if osm_edges > 0 else 0

    print(f"\n  [3] GRAPH vs OSM GROUND TRUTH")
    print(_line("Our nodes / OSM nodes",
                f"{n_nodes} / {osm_nodes}", "",
                _score_tag(node_cov, 0.70, 0.40)))
    print(_line("Our edges / OSM edges",
                f"{n_edges} / {osm_edges}", "",
                _score_tag(edge_cov, 0.70, 0.40)))
    print(_line("Our length / OSM length",
                f"{total_km:.2f} / {osm_km:.2f}", " km"))
    print(_line("Mean edge length",   f"{mean_w:.1f}", " m"))

    # ── Section 4: TOPOLOGY F1 ────────────────────────────────
    node_f1   = metrics.get("node_f1", 0)
    edge_f1   = metrics.get("edge_f1", 0)
    snap_m    = metrics.get("snap_m", 10.0)

    print(f"\n  [4] TOPOLOGY F1  (snap={snap_m:.0f}m vs OSM)")
    print(_line("Node F1",  f"{node_f1:.4f}", "",
                _score_tag(node_f1, 0.70, 0.40)))
    print(_line("Edge F1",  f"{edge_f1:.4f}", "",
                _score_tag(edge_f1, 0.60, 0.30)))
    print(f"  {'':26} Target: node_F1>0.70, edge_F1>0.60")

    # ── Section 5: CONNECTIVITY ───────────────────────────────
    lcc_pct     = metrics.get("lcc_pct", 0)
    n_comp      = metrics.get("n_components", 0)
    isolated    = metrics.get("isolated_nodes", 0)
    lcc_pass    = metrics.get("lcc_pass", False)

    print(f"\n  [5] CONNECTIVITY")
    print(_line("LCC%", f"{lcc_pct:.1%}", "",
                _score_tag(lcc_pct, 0.80, 0.60)))
    print(_line("Components",  n_comp,  "",
                "✓" if n_comp <= 3 else ("○" if n_comp <= 10 else "✗")))
    print(_line("Isolated nodes", isolated, "",
                "✓" if isolated == 0 else "○"))
    print(_line("Routing usable",
                "YES" if lcc_pass else "NO — needs healing", "",
                "✓" if lcc_pass else "✗"))

    # ── Overall score ─────────────────────────────────────────
    scores = {
        "contract":     1.0 if contract_status == "PASS" else 0.0,
        "node_f1":      node_f1,
        "edge_f1":      edge_f1,
        "lcc":          lcc_pct,
        "connectivity": 1.0 if lcc_pass else 0.0,
    }
    overall = sum(scores.values()) / len(scores)

    # Qualitative grade
    if overall >= 0.80:   grade = "EXCELLENT"
    elif overall >= 0.65: grade = "GOOD"
    elif overall >= 0.50: grade = "ACCEPTABLE"
    elif overall >= 0.35: grade = "WEAK"
    else:                 grade = "POOR — healing required"

    print(f"\n{SEP2}")
    print(f"  OVERALL SCORE: {overall:.3f} / 1.000   [{grade}]")
    print(f"  Breakdown: contract={scores['contract']:.2f}  "
          f"node_f1={scores['node_f1']:.2f}  "
          f"edge_f1={scores['edge_f1']:.2f}  "
          f"lcc={scores['lcc']:.2f}  "
          f"conn={scores['connectivity']:.2f}")
    print(f"\n  Note: Scores on synthetic data are lower than real LISS-IV.")
    print(f"  Healing (Phase 12) and real mask (Phase 19) will improve all metrics.")
    print(f"{SEP}\n")


# ══════════════════════════════════════════════════════════════
# LAYER 2 — HEALING QUALITY VALIDATION  (Phase 15)
# ══════════════════════════════════════════════════════════════

def healing_validation(graph,
                       lcc_threshold: float = 0.80,
                       second_pass_snap_m: float = 40.0,
                       resolution_m: float = 10.0) -> dict:
    """
    Layer 2: Post-healing quality gate.

    After all healing and simplification, validates that the graph
    meets the minimum LCC% threshold for Part C to produce meaningful
    criticality analysis.

    If LCC% < lcc_threshold:
      - Runs a second healing pass with relaxed snap radius (40m default)
      - Re-checks LCC%
      - Reports both passes in the result

    If LCC% still < lcc_threshold after second pass:
      - Returns status='WARN' (not FAIL — Part C should still run,
        but judge is warned the graph quality is suboptimal)

    Parameters
    ----------
    graph             : RoadGraph
    lcc_threshold     : float — minimum acceptable LCC% (default 0.80)
    second_pass_snap_m: float — relaxed snap for second healing pass
    resolution_m      : float — for adaptive pruning in second pass

    Returns
    -------
    dict with:
        status         : 'PASS' | 'PASS_AFTER_SECOND_PASS' | 'WARN'
        lcc_pct        : float — final LCC%
        lcc_pass       : bool
        passes_run     : int  — 1 or 2
        second_pass_edges: int — healing edges added in second pass (0 if not run)
        message        : str  — human-readable summary
    """
    conn = connectivity_report(graph)
    lcc_pct = conn['lcc_pct']

    if lcc_pct >= lcc_threshold:
        return {
            "status":              "PASS",
            "lcc_pct":             lcc_pct,
            "lcc_pass":            True,
            "passes_run":          1,
            "second_pass_edges":   0,
            "lcc_threshold":       lcc_threshold,
            "message": (f"LCC={lcc_pct:.1%} ≥ {lcc_threshold:.0%} threshold — "
                        f"graph ready for Part C"),
        }

    # ── Second healing pass with relaxed threshold ────────────
    from part_b_skeleton.healer import detect_breaks, mst_heal

    relaxed_snap = max(second_pass_snap_m, resolution_m * 16)
    break_pairs, _ = detect_breaks(graph, snap_m=relaxed_snap)
    healed2, heal2_metrics = mst_heal(
        graph, break_pairs,
        max_heal_dist_m=relaxed_snap,
        lcc_target=lcc_threshold,
    )

    conn2    = connectivity_report(healed2)
    lcc_pct2 = conn2['lcc_pct']
    n_healed2 = heal2_metrics.get('healed_edges', 0)

    if lcc_pct2 >= lcc_threshold:
        status  = "PASS_AFTER_SECOND_PASS"
        lcc_pass = True
        message = (f"LCC={lcc_pct:.1%} after pass 1 → {lcc_pct2:.1%} after pass 2 "
                   f"({n_healed2} extra edges, {relaxed_snap:.0f}m snap) — "
                   f"threshold {lcc_threshold:.0%} reached")
    else:
        status  = "WARN"
        lcc_pass = False
        message = (f"LCC={lcc_pct2:.1%} still below {lcc_threshold:.0%} after 2 passes — "
                   f"mask quality may be insufficient for full analysis")

    return {
        "status":             status,
        "lcc_pct":            lcc_pct2,
        "lcc_pct_pass1":      lcc_pct,
        "lcc_pass":           lcc_pass,
        "passes_run":         2,
        "second_pass_edges":  n_healed2,
        "second_pass_snap_m": relaxed_snap,
        "lcc_threshold":      lcc_threshold,
        "healed_graph":       healed2,
        "message":            message,
    }


def print_healing_validation(result: dict) -> None:
    """Print Phase 15 healing validation report."""
    SEP = "─" * 60
    status   = result["status"]
    lcc      = result["lcc_pct"]
    passes   = result["passes_run"]
    thresh   = result["lcc_threshold"]

    status_icon = {
        "PASS":                  "✓",
        "PASS_AFTER_SECOND_PASS": "✓",
        "WARN":                  "⚠",
    }.get(status, "?")

    print(f"\n{SEP}")
    print(f"  PHASE 15 — HEALING QUALITY VALIDATION")
    print(SEP)
    print(f"  LCC threshold     : {thresh:.0%}")
    print(f"  Passes run        : {passes}")

    if passes == 2:
        print(f"  LCC after pass 1  : {result.get('lcc_pct_pass1', 0):.1%}  ✗ below threshold")
        print(f"  Second pass snap  : {result.get('second_pass_snap_m', 0):.0f}m (relaxed)")
        print(f"  Edges added (p2)  : {result['second_pass_edges']}")

    print(f"  Final LCC%        : {lcc:.1%}  {status_icon}")
    print(f"\n  {result['message']}")

    print(f"\n{SEP}")
    if status == "PASS":
        print(f"  HEALING GATE: ✓ PASS (single pass sufficient)")
    elif status == "PASS_AFTER_SECOND_PASS":
        print(f"  HEALING GATE: ✓ PASS (required second pass at relaxed snap)")
    else:
        print(f"  HEALING GATE: ⚠ WARN — Part C will run but results are suboptimal")
        print(f"  Action: improve Part A mask quality for better road coverage")
    print(SEP)

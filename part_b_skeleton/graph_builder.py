"""
part_b_skeleton/graph_builder.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 05: sknw Graph Extraction + RoadGraph Emission

Responsibilities:
  1. Run sknw.build_sknw() on the boolean skeleton
  2. Convert every pixel (row, col) → (lat, lon) via AffineTransform
  3. Compute Haversine weight_m for every edge
  4. Serialize to RoadGraph dataclass → graph.json
  5. Emit basic graph stats for the judge report

sknw data structure (important):
  node data: {'o': array([row, col])}   ← centroid in pixel coords
  edge data: {'pts': array([[row,col],...])}  ← polyline in pixel coords
  All coordinates are (row, col) order — we convert to (lat, lon).

Why Haversine for weight_m:
  Euclidean pixel distance is wrong at lat/lon — at 13°N,
  1° lon ≈ 97.6km but 1° lat ≈ 111.3km. Using Euclidean
  distances would corrupt routing by up to 14%.
  Haversine gives true geodesic distance along the polyline.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
import math
import os
import sys
from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np
import networkx as nx
import sknw

# ── repo root on path ─────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from shared.schema import GraphEdge, GraphNode, RoadGraph
from shared.config import TARGET_CRS
from part_b_skeleton.loader import AffineTransform


# ══════════════════════════════════════════════════════════════
# HAVERSINE DISTANCE
# ══════════════════════════════════════════════════════════════

def haversine_m(lat1: float, lon1: float,
                lat2: float, lon2: float) -> float:
    """
    Haversine distance between two (lat, lon) points in metres.
    Accurate to within 0.5% for distances under 100km.
    """
    R = 6_371_000.0  # Earth radius in metres
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def haversine_polyline_m(points: List[Tuple[float, float]]) -> float:
    """
    Total Haversine length of a polyline [(lat,lon), ...] in metres.
    Sums segment distances — correct for curved roads.

    Returns 0.0 for single-point or empty polylines.
    """
    if len(points) < 2:
        return 0.0
    total = 0.0
    for i in range(len(points) - 1):
        total += haversine_m(points[i][0], points[i][1],
                             points[i+1][0], points[i+1][1])
    return total


# ══════════════════════════════════════════════════════════════
# sknw → RoadGraph CONVERSION
# ══════════════════════════════════════════════════════════════

def sknw_to_road_graph(sknw_graph: nx.Graph,
                       affine: AffineTransform) -> RoadGraph:
    """
    Convert a sknw NetworkX graph (pixel coords) to a RoadGraph (lat/lon).

    sknw graph structure:
      node[n]['o']   = array([row, col])       ← node centroid
      edge[u][v]['pts'] = array([[row,col],...]) ← edge polyline

    Steps:
      1. Convert node centroids from (row,col) → (lat,lon)
      2. For each edge, convert pts polyline to lat/lon
      3. Ensure polyline starts at source node and ends at target node
      4. Compute Haversine weight_m along the polyline
      5. Assign sequential integer IDs

    Parameters
    ----------
    sknw_graph : nx.Graph — output of sknw.build_sknw()
    affine     : AffineTransform — from Phase 03

    Returns
    -------
    RoadGraph with nodes, edges, crs='EPSG:4326'
    """
    # ── Step 1: Build node ID mapping ────────────────────────
    # sknw uses arbitrary integer node IDs — remap to sequential 0..N-1
    sknw_ids = list(sknw_graph.nodes())
    id_map   = {old_id: new_id for new_id, old_id in enumerate(sknw_ids)}

    # ── Step 2: Convert nodes ─────────────────────────────────
    nodes: List[GraphNode] = []
    for sknw_id in sknw_ids:
        node_data = sknw_graph.nodes[sknw_id]
        centroid  = node_data['o']          # array([row, col])
        row, col  = float(centroid[0]), float(centroid[1])
        lat, lon  = affine.pixel_to_latlon(row, col)

        nodes.append(GraphNode(
            id  = id_map[sknw_id],
            lat = round(lat, 8),
            lon = round(lon, 8),
        ))

    # ── Step 3: Convert edges ─────────────────────────────────
    edges: List[GraphEdge] = []
    for u, v, edge_data in sknw_graph.edges(data=True):
        src_id = id_map[u]
        tgt_id = id_map[v]

        # Edge geometry: sknw 'pts' is array of [row, col] pixel coords
        pts = edge_data.get('pts', np.array([]))

        if len(pts) == 0:
            # Degenerate edge — no intermediate points
            # Use the two node centroids as geometry endpoints
            src_node = nodes[src_id]
            tgt_node = nodes[tgt_id]
            geom = [
                (src_node.lat, src_node.lon),
                (tgt_node.lat, tgt_node.lon),
            ]
        else:
            # Convert pixel polyline to lat/lon
            geom_latlon = []
            for pt in pts:
                row_p, col_p = float(pt[0]), float(pt[1])
                lat_p, lon_p = affine.pixel_to_latlon(row_p, col_p)
                geom_latlon.append((round(lat_p, 8), round(lon_p, 8)))

            # sknw pts may not include the endpoint nodes themselves —
            # prepend source node and append target node to close the polyline
            src_node = nodes[src_id]
            tgt_node = nodes[tgt_id]
            src_pt   = (src_node.lat, src_node.lon)
            tgt_pt   = (tgt_node.lat, tgt_node.lon)

            # Only prepend/append if not already present (within 1e-6°)
            def _same_pt(a, b):
                return abs(a[0] - b[0]) < 1e-6 and abs(a[1] - b[1]) < 1e-6

            if not _same_pt(geom_latlon[0], src_pt):
                geom_latlon = [src_pt] + geom_latlon
            if not _same_pt(geom_latlon[-1], tgt_pt):
                geom_latlon = geom_latlon + [tgt_pt]

            geom = geom_latlon

        # Compute true Haversine length along polyline
        weight_m = haversine_polyline_m(geom)

        # Guard: weight_m must be > 0
        if weight_m <= 0.0:
            # Fallback: straight-line Haversine between endpoints
            weight_m = haversine_m(
                geom[0][0], geom[0][1],
                geom[-1][0], geom[-1][1]
            )
        if weight_m <= 0.0:
            weight_m = affine.resolution_m  # absolute fallback: 1 pixel length

        edges.append(GraphEdge(
            source   = src_id,
            target   = tgt_id,
            weight_m = round(weight_m, 3),
            geometry = [list(pt) for pt in geom],
        ))

    return RoadGraph(
        nodes = nodes,
        edges = edges,
        crs   = TARGET_CRS,
    )


# ══════════════════════════════════════════════════════════════
# SERIALIZATION
# ══════════════════════════════════════════════════════════════

def road_graph_to_dict(graph: RoadGraph) -> dict:
    """
    Serialize RoadGraph to a JSON-compatible dict.
    Matches shared/schema.py exactly.
    """
    return {
        "crs": graph.crs,
        "nodes": [
            {"id": n.id, "lat": n.lat, "lon": n.lon}
            for n in graph.nodes
        ],
        "edges": [
            {
                "source":   e.source,
                "target":   e.target,
                "weight_m": e.weight_m,
                "geometry": e.geometry,
            }
            for e in graph.edges
        ],
    }


def save_graph_json(graph: RoadGraph, output_path: str) -> None:
    """
    Write RoadGraph to graph.json.
    Creates parent directory if needed.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data = road_graph_to_dict(graph)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


# ══════════════════════════════════════════════════════════════
# GRAPH STATS
# ══════════════════════════════════════════════════════════════

def compute_graph_stats(graph: RoadGraph) -> Dict:
    """
    Compute basic graph statistics for the judge report.
    """
    n_nodes = len(graph.nodes)
    n_edges = len(graph.edges)

    if n_edges == 0:
        return {
            "n_nodes": n_nodes, "n_edges": 0,
            "mean_weight_m": 0, "min_weight_m": 0,
            "max_weight_m": 0, "total_length_km": 0,
            "mean_geometry_points": 0,
        }

    weights = [e.weight_m for e in graph.edges]
    geom_lens = [len(e.geometry) for e in graph.edges]

    return {
        "n_nodes":             n_nodes,
        "n_edges":             n_edges,
        "mean_weight_m":       round(float(np.mean(weights)), 2),
        "min_weight_m":        round(float(np.min(weights)), 2),
        "max_weight_m":        round(float(np.max(weights)), 2),
        "total_length_km":     round(sum(weights) / 1000, 3),
        "mean_geometry_points": round(float(np.mean(geom_lens)), 1),
    }


def validate_graph_basic(graph: RoadGraph) -> list:
    """
    Quick sanity checks on the RoadGraph before saving.
    Returns list of violation strings.
    """
    violations = []

    if len(graph.nodes) == 0:
        violations.append("RoadGraph has zero nodes")
    if len(graph.edges) == 0:
        violations.append("RoadGraph has zero edges")
    if graph.crs != TARGET_CRS:
        violations.append(f"crs='{graph.crs}' must be '{TARGET_CRS}'")

    # Check all edge endpoints reference valid node IDs
    valid_ids = {n.id for n in graph.nodes}
    for i, e in enumerate(graph.edges):
        if e.source not in valid_ids:
            violations.append(f"edges[{i}].source={e.source} not in node IDs")
        if e.target not in valid_ids:
            violations.append(f"edges[{i}].target={e.target} not in node IDs")
        if e.weight_m <= 0:
            violations.append(f"edges[{i}].weight_m={e.weight_m} must be > 0")
        if len(e.geometry) < 2:
            violations.append(f"edges[{i}].geometry has < 2 points")

    return violations


# ══════════════════════════════════════════════════════════════
# PRINT REPORT
# ══════════════════════════════════════════════════════════════

def print_graph_report(stats: Dict, violations: list,
                       output_path: str) -> None:
    """Print Phase 05 graph extraction report."""
    SEP = "─" * 60
    print(f"\n{SEP}")
    print(f"  PHASE 05 — GRAPH EXTRACTION & EMISSION")
    print(SEP)
    print(f"  Nodes             : {stats['n_nodes']}")
    print(f"  Edges             : {stats['n_edges']}")
    print(f"  Mean edge length  : {stats['mean_weight_m']:.1f} m")
    print(f"  Min  edge length  : {stats['min_weight_m']:.1f} m")
    print(f"  Max  edge length  : {stats['max_weight_m']:.1f} m")
    print(f"  Total road length : {stats['total_length_km']:.3f} km")
    print(f"  Mean geom points  : {stats['mean_geometry_points']:.1f} pts/edge")
    print(f"  CRS               : EPSG:4326 [OK]")
    print(f"  Output            : {output_path}")

    if violations:
        print(f"\n  Violations ({len(violations)}):")
        for v in violations:
            print(f"    ✗ {v}")
        print(f"\n{SEP}")
        print(f"  GRAPH: ✗ FAIL")
    else:
        print(f"\n{SEP}")
        print(f"  GRAPH: [OK] PASS — graph.json emitted, ready for Part C")
    print(SEP)


# ══════════════════════════════════════════════════════════════
# TOP-LEVEL FUNCTION (called by run.py)
# ══════════════════════════════════════════════════════════════

def build_and_save_graph(skeleton: np.ndarray,
                         affine: AffineTransform,
                         output_path: str
                         ) -> Tuple[RoadGraph, Dict, list]:
    """
    Full Phase 05 pipeline:
      skeleton → sknw → RoadGraph → graph.json

    Parameters
    ----------
    skeleton    : np.ndarray bool, shape (H, W) — from Phase 04
    affine      : AffineTransform — from Phase 03
    output_path : str — where to write graph.json

    Returns
    -------
    (road_graph, stats, violations)
    """
    # ── sknw extraction ───────────────────────────────────────
    sknw_graph = sknw.build_sknw(skeleton.astype(np.uint16))

    # ── Convert to RoadGraph ──────────────────────────────────
    road_graph = sknw_to_road_graph(sknw_graph, affine)

    # ── Validate ──────────────────────────────────────────────
    violations = validate_graph_basic(road_graph)

    # ── Stats ─────────────────────────────────────────────────
    stats = compute_graph_stats(road_graph)

    # ── Save ──────────────────────────────────────────────────
    if not violations:
        save_graph_json(road_graph, output_path)

    # ── Report ────────────────────────────────────────────────
    print_graph_report(stats, violations, output_path)

    return road_graph, stats, violations

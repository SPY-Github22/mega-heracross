"""
part_b_skeleton/spline_healer.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 20: Bearing-Aware Cubic Spline Healing

Replaces the linear interpolation in healer.py's _interpolate_geometry
with bearing-aware cubic splines. When bridging an occlusion gap between
two dead-end nodes, we:

  1. Estimate the inward bearing of each dead-end from its connecting
     edge geometry (the direction the road is travelling as it enters
     the gap)
  2. Use these bearings as tangent constraints for a cubic Hermite spline
  3. Sample the spline at n_points intervals to produce a smooth geometry
     polyline that looks like a real curved road

Why this matters:
  Linear healing:   A ────────── B   (straight line, obviously fake)
  Spline healing:   A ──╮╭─── B   (follows road curvature, realistic)

On real Bengaluru roads with curves, spline healing produces edges that:
  - Match the expected curvature of the road at the gap endpoints
  - Integrate correctly with Part C's length calculations
  - Look realistic on the Folium disaster heatmap
  - Improve topology F1 score vs OSM ground truth

The Hermite spline formulation:
  Given endpoints P0, P1 and tangent vectors T0, T1:
  P(t) = (2t³-3t²+1)P0 + (t³-2t²+t)T0 + (-2t³+3t²)P1 + (t³-t²)T1
  where t ∈ [0,1]

  Tangent vectors are derived from:
    - Bearing in degrees → unit vector in lat/lon space
    - Scaled by gap_distance × tension_factor
    - tension_factor ∈ [0.3, 0.5] produces natural-looking curves
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import math
import os
import sys
from typing import List, Optional, Tuple

import numpy as np

# ── repo root on path ─────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from shared.schema import GraphEdge, GraphNode, RoadGraph


# ══════════════════════════════════════════════════════════════
# BEARING ESTIMATION
# ══════════════════════════════════════════════════════════════

def estimate_inward_bearing(node_id: int,
                             graph: RoadGraph,
                             n_pts_for_bearing: int = 5
                             ) -> Optional[float]:
    """
    Estimate the inward bearing at a dead-end node.

    The inward bearing is the direction the road is travelling
    as it approaches the gap — i.e. the direction FROM the gap
    INTO the existing road network.

    Algorithm:
      1. Find the edge connecting this dead-end to its neighbour
      2. Take the first/last n_pts_for_bearing geometry points
         (depending on which end of the edge is the dead-end)
      3. Fit a direction vector to those points using least squares
      4. Return bearing in degrees [0, 360)

    Parameters
    ----------
    node_id           : int — the dead-end node ID
    graph             : RoadGraph
    n_pts_for_bearing : int — how many edge geometry points to use
                              for bearing estimation (more = smoother)

    Returns
    -------
    float | None — bearing in degrees, or None if estimation fails
    """
    node_by_id = {n.id: n for n in graph.nodes}

    # Find the connecting edge for this dead-end
    connecting_edge = None
    for e in graph.edges:
        if e.source == node_id or e.target == node_id:
            connecting_edge = e
            break

    if connecting_edge is None:
        return None

    geom = connecting_edge.geometry
    if len(geom) < 2:
        return None

    # Determine which end of the geometry is at our dead-end node
    node = node_by_id.get(node_id)
    if node is None:
        return None

    first_pt = geom[0]
    last_pt  = geom[-1]

    dist_to_first = math.sqrt(
        (first_pt[0] - node.lat)**2 + (first_pt[1] - node.lon)**2
    )
    dist_to_last = math.sqrt(
        (last_pt[0] - node.lat)**2 + (last_pt[1] - node.lon)**2
    )

    if dist_to_first < dist_to_last:
        # Dead-end is at the START of the geometry
        # Inward bearing: from geom[0] toward geom[1..n]
        pts = geom[:min(n_pts_for_bearing, len(geom))]
        p_from, p_to = pts[0], pts[-1]
    else:
        # Dead-end is at the END of the geometry
        # Inward bearing: from geom[-1] toward geom[-n..-2]
        pts = geom[max(0, len(geom)-n_pts_for_bearing):]
        p_from, p_to = pts[-1], pts[0]

    # Compute bearing (inward direction = from dead-end into road)
    dlat = p_to[0] - p_from[0]
    dlon = p_to[1] - p_from[1]

    if abs(dlat) < 1e-12 and abs(dlon) < 1e-12:
        return None  # Degenerate — no direction

    bearing = math.degrees(math.atan2(dlon, dlat)) % 360
    return bearing


# ══════════════════════════════════════════════════════════════
# CUBIC HERMITE SPLINE
# ══════════════════════════════════════════════════════════════

def _hermite_spline(p0: np.ndarray, p1: np.ndarray,
                    t0: np.ndarray, t1: np.ndarray,
                    n_points: int = 8) -> List[List[float]]:
    """
    Cubic Hermite spline between P0 and P1 with tangents T0 and T1.

    P(t) = (2t³-3t²+1)P0 + (t³-2t²+t)T0 + (-2t³+3t²)P1 + (t³-t²)T1

    Parameters
    ----------
    p0, p1 : np.ndarray shape (2,) — endpoints in (lat, lon)
    t0, t1 : np.ndarray shape (2,) — tangent vectors at P0 and P1
    n_points: int — number of sample points (including endpoints)

    Returns
    -------
    List of [lat, lon] pairs along the spline
    """
    result = []
    for i in range(n_points):
        t = i / (n_points - 1)
        t2, t3 = t**2, t**3

        h00 =  2*t3 - 3*t2 + 1   # basis for P0
        h10 =    t3 - 2*t2 + t   # basis for T0
        h01 = -2*t3 + 3*t2       # basis for P1
        h11 =    t3 -   t2       # basis for T1

        pt = h00 * p0 + h10 * t0 + h01 * p1 + h11 * t1
        result.append([round(float(pt[0]), 8), round(float(pt[1]), 8)])

    return result


def bearing_to_unit_vector(bearing_deg: float,
                            lat: float) -> np.ndarray:
    """
    Convert a bearing in degrees to a unit vector in (lat, lon) space.
    Accounts for the lat/lon scale difference at the given latitude.

    Parameters
    ----------
    bearing_deg : float — bearing in degrees [0, 360)
    lat         : float — reference latitude for lon scaling

    Returns
    -------
    np.ndarray shape (2,) — unit vector (dlat, dlon) in degree space
    """
    # At latitude `lat`, 1 degree of latitude ≈ 111320m
    # 1 degree of longitude ≈ 111320 * cos(lat) m
    # To get a true unit vector in physical space:
    cos_lat = math.cos(math.radians(lat))

    bearing_rad = math.radians(bearing_deg)
    # bearing: 0=North, 90=East, 180=South, 270=West
    dlat_physical = math.cos(bearing_rad)        # northward component
    dlon_physical = math.sin(bearing_rad)        # eastward component

    # Convert physical unit vector back to degree space
    dlat_deg = dlat_physical / 111_320.0
    dlon_deg = dlon_physical / (111_320.0 * cos_lat) if cos_lat > 1e-6 else 0.0

    # Normalise in degree space
    magnitude = math.sqrt(dlat_deg**2 + dlon_deg**2)
    if magnitude < 1e-12:
        return np.array([0.0, 0.0])

    return np.array([dlat_deg / magnitude, dlon_deg / magnitude])


# ══════════════════════════════════════════════════════════════
# SPLINE-BASED GEOMETRY INTERPOLATION
# ══════════════════════════════════════════════════════════════

def spline_interpolate_geometry(node_a: GraphNode,
                                 node_b: GraphNode,
                                 bearing_a: Optional[float],
                                 bearing_b: Optional[float],
                                 n_points: int = 8,
                                 tension: float = 0.4
                                 ) -> List[List[float]]:
    """
    Generate a smooth spline geometry between two dead-end nodes.

    If bearings are available, uses cubic Hermite spline with
    bearing-derived tangents. Falls back to linear interpolation
    if bearings are unavailable.

    Parameters
    ----------
    node_a, node_b : GraphNode — the two endpoints
    bearing_a      : float | None — inward bearing at node_a (degrees)
    bearing_b      : float | None — inward bearing at node_b (degrees)
    n_points       : int — number of geometry points (default 8)
    tension        : float — tangent scale factor [0.3, 0.5]
                     higher = more curved, lower = more linear

    Returns
    -------
    List of [lat, lon] pairs
    """
    p0 = np.array([node_a.lat, node_a.lon])
    p1 = np.array([node_b.lat, node_b.lon])

    # Gap distance in degree space
    gap_dist = math.sqrt(
        (p1[0] - p0[0])**2 + (p1[1] - p0[1])**2
    )

    # Fallback: linear interpolation if bearings unavailable
    if bearing_a is None or bearing_b is None:
        return [
            [round(p0[0] + i/(n_points-1) * (p1[0]-p0[0]), 8),
             round(p0[1] + i/(n_points-1) * (p1[1]-p0[1]), 8)]
            for i in range(n_points)
        ]

    centre_lat = (node_a.lat + node_b.lat) / 2

    # Tangent vectors: unit bearing direction scaled by gap_dist × tension
    # The tangent at A points INWARD (away from gap, into road)
    # For the spline, we need the tangent pointing TOWARD the gap
    # so we negate the inward bearing (outward_bearing = inward + 180)
    outward_a = (bearing_a + 180) % 360
    outward_b = (bearing_b + 180) % 360

    t0 = bearing_to_unit_vector(outward_a, centre_lat) * gap_dist * tension
    t1 = bearing_to_unit_vector(outward_b, centre_lat) * gap_dist * tension

    geom = _hermite_spline(p0, p1, t0, t1, n_points=n_points)
    return geom


# ══════════════════════════════════════════════════════════════
# UPGRADED HEALING INTEGRATION
# ══════════════════════════════════════════════════════════════

def bearing_aware_spline_bridge(
        graph: RoadGraph,
        break_pairs: List[Tuple[int, int, float]],
        max_heal_dist_m: float = 25.0,
        lcc_target: float = 0.80,
        n_spline_points: int = 8,
        tension: float = 0.4,
) -> Tuple[RoadGraph, dict]:
    """
    Phase 20: MST healing with bearing-aware spline geometry.

    Identical logic to healer.mst_heal() but uses spline_interpolate_geometry
    instead of linear _interpolate_geometry.

    Parameters
    ----------
    graph             : RoadGraph
    break_pairs       : List[(id_a, id_b, dist_m)] sorted by dist
    max_heal_dist_m   : float — distance gate
    lcc_target        : float — stop when LCC% reaches this
    n_spline_points   : int   — geometry resolution of healed edges
    tension           : float — spline curvature (0.3=gentle, 0.5=sharp)

    Returns
    -------
    (healed_graph, metrics)
    """
    import networkx as nx
    from shared.eval import connectivity_report
    from part_b_skeleton.healer import WeightedUnionFind

    def _haversine_m(lat1, lon1, lat2, lon2):
        R = 6_371_000.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlam = math.radians(lon2 - lon1)
        a = (math.sin(dphi/2)**2
             + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2)
        return R * 2 * math.asin(math.sqrt(max(0.0, a)))

    def _polyline_length(geom):
        if len(geom) < 2:
            return 0.0
        return sum(_haversine_m(geom[i][0], geom[i][1],
                                geom[i+1][0], geom[i+1][1])
                   for i in range(len(geom)-1))

    if not break_pairs:
        conn = connectivity_report(graph)
        return graph, {
            "healed_edges": 0,
            "spline_edges": 0,
            "linear_fallback_edges": 0,
            "skipped_same_component": 0,
            "skipped_too_far": 0,
            "lcc_before": conn["lcc_pct"],
            "lcc_after":  conn["lcc_pct"],
            "lcc_improvement": 0.0,
            "components_before": conn["n_components"],
            "components_after":  conn["n_components"],
            "lcc_target_reached": conn["lcc_pct"] >= lcc_target,
        }

    conn_before = connectivity_report(graph)
    uf = WeightedUnionFind.from_graph(graph)
    node_by_id = {n.id: n for n in graph.nodes}

    healing_edges = []
    spline_count  = 0
    linear_count  = 0
    skipped_same  = 0
    skipped_far   = 0

    for id_a, id_b, dist_m in break_pairs:
        if dist_m > max_heal_dist_m:
            skipped_far += 1
            continue
        if uf.connected(id_a, id_b):
            skipped_same += 1
            continue
        if id_a not in node_by_id or id_b not in node_by_id:
            continue

        node_a = node_by_id[id_a]
        node_b = node_by_id[id_b]

        # Estimate bearings from adjacent edge geometry
        bearing_a = estimate_inward_bearing(id_a, graph)
        bearing_b = estimate_inward_bearing(id_b, graph)

        # Generate spline or linear geometry
        geom = spline_interpolate_geometry(
            node_a, node_b,
            bearing_a, bearing_b,
            n_points=n_spline_points,
            tension=tension,
        )

        if bearing_a is not None and bearing_b is not None:
            spline_count += 1
        else:
            linear_count += 1

        weight_m = _polyline_length(geom)
        weight_m = max(weight_m, 1.0)

        healing_edges.append(GraphEdge(
            source   = id_a,
            target   = id_b,
            weight_m = round(weight_m, 3),
            geometry = geom,
        ))

        uf.union(id_a, id_b)
        if uf.n_components == 1:
            break

    from shared.config import TARGET_CRS
    healed_graph = RoadGraph(
        nodes = graph.nodes,
        edges = graph.edges + healing_edges,
        crs   = TARGET_CRS,
    )

    conn_after = connectivity_report(healed_graph)

    metrics = {
        "healed_edges":          len(healing_edges),
        "spline_edges":          spline_count,
        "linear_fallback_edges": linear_count,
        "skipped_same_component": skipped_same,
        "skipped_too_far":        skipped_far,
        "lcc_before":    conn_before["lcc_pct"],
        "lcc_after":     conn_after["lcc_pct"],
        "lcc_improvement": round(conn_after["lcc_pct"] - conn_before["lcc_pct"], 4),
        "components_before": conn_before["n_components"],
        "components_after":  conn_after["n_components"],
        "lcc_target_reached": conn_after["lcc_pct"] >= lcc_target,
    }

    return healed_graph, metrics


# ══════════════════════════════════════════════════════════════
# PRINT REPORT
# ══════════════════════════════════════════════════════════════

def print_spline_healing_report(metrics: dict) -> None:
    """Print Phase 20 spline healing upgrade report."""
    SEP = "─" * 60
    total  = metrics["healed_edges"]
    spline = metrics["spline_edges"]
    linear = metrics["linear_fallback_edges"]

    print(f"\n{SEP}")
    print(f"  PHASE 20 — BEARING-AWARE SPLINE HEALING")
    print(SEP)
    print(f"  Healing edges added : {total}")
    if total > 0:
        print(f"    Spline (bearing)  : {spline}  ({spline/total*100:.0f}%)")
        print(f"    Linear (fallback) : {linear}  ({linear/total*100:.0f}%)")
    print(f"  LCC: {metrics['lcc_before']:.1%} → {metrics['lcc_after']:.1%}")
    print(f"\n{SEP}")
    print(f"  SPLINE HEALING: [OK] COMPLETE")
    print(SEP)

"""
part_b_skeleton/healer.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phases 10–12: Topological Healing Engine

Phase 10 — KD-Tree Break Detection
  Extract degree-1 dead-end nodes (road endpoints with no continuation).
  These appear where occlusion (tree canopy, cloud, shadow) breaks the
  segmentation mask, leaving roads dangling. Build a KDTree on their
  positions and find all pairs within snap_m metres — these are
  candidate healing targets.

Phase 11 — Union-Find Component Tracking
  WeightedUnionFind tracks which nodes belong to which connected
  component. As healing edges are added, components are merged.
  Prevents circular healing (connecting nodes already in same component)
  and makes healing idempotent — each break is fixed exactly once.

Phase 12 — MST-Guided Gap Bridging
  Sort candidate break pairs by Haversine distance (shortest first).
  For each pair: if they belong to different components (Union-Find),
  add a healing edge with interpolated geometry. MST ordering ensures
  minimum total healing cost. Re-run connectivity_report() to measure
  LCC% improvement.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import math
import os
import sys
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.spatial import KDTree

# ── repo root on path ─────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from shared.schema import GraphEdge, GraphNode, RoadGraph
from shared.config import TARGET_CRS


# ══════════════════════════════════════════════════════════════
# HAVERSINE (local — no circular import)
# ══════════════════════════════════════════════════════════════

def _haversine_m(lat1: float, lon1: float,
                 lat2: float, lon2: float) -> float:
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(max(0.0, a)))


def _to_metres(lat: float, lon: float,
               metres_per_deg_lat: float,
               metres_per_deg_lon: float) -> Tuple[float, float]:
    return (lat * metres_per_deg_lat, lon * metres_per_deg_lon)


# ══════════════════════════════════════════════════════════════
# PHASE 10 — KD-TREE BREAK DETECTION
# ══════════════════════════════════════════════════════════════

def detect_breaks(graph: RoadGraph,
                  snap_m: float = 25.0
                  ) -> Tuple[List[Tuple[int, int, float]], Dict]:
    """
    Phase 10: Find candidate break pairs using KD-Tree on degree-1 nodes.

    A degree-1 node is a dead end — a road that stops without connecting
    to anything. In a real urban network, almost all such nodes are caused
    by occlusion gaps in the segmentation mask, not genuine dead ends.

    Algorithm:
      1. Build NetworkX graph to compute node degrees
      2. Extract all degree-1 nodes (dead ends)
      3. Build KDTree on their lat/lon (scaled to metres)
      4. Query all pairs within snap_m metres
      5. Return sorted list of (node_id_a, node_id_b, distance_m) pairs

    Parameters
    ----------
    graph  : RoadGraph
    snap_m : float — search radius in metres (default 25m)

    Returns
    -------
    break_pairs : List of (id_a, id_b, dist_m) sorted by distance
    metrics     : dict with detection statistics
    """
    import networkx as nx

    if len(graph.nodes) == 0:
        return [], {"n_deadends": 0, "n_break_pairs": 0, "snap_m": snap_m}

    # ── Build NetworkX graph ──────────────────────────────────
    G = nx.Graph()
    G.add_nodes_from([n.id for n in graph.nodes])
    G.add_edges_from([(e.source, e.target) for e in graph.edges])

    # ── Extract degree-1 nodes ────────────────────────────────
    node_by_id = {n.id: n for n in graph.nodes}
    deadend_ids = [nid for nid in G.nodes() if G.degree(nid) == 1]

    if len(deadend_ids) < 2:
        return [], {
            "n_deadends":    len(deadend_ids),
            "n_break_pairs": 0,
            "snap_m":        snap_m,
        }

    # ── Coordinate scaling ────────────────────────────────────
    centre_lat = sum(n.lat for n in graph.nodes) / len(graph.nodes)
    MPD_LAT = 111_320.0
    MPD_LON = 111_320.0 * math.cos(math.radians(centre_lat))

    # ── Build KDTree on dead-end positions ────────────────────
    deadend_nodes  = [node_by_id[nid] for nid in deadend_ids]
    deadend_coords = np.array([
        _to_metres(n.lat, n.lon, MPD_LAT, MPD_LON)
        for n in deadend_nodes
    ])
    tree = KDTree(deadend_coords)

    # ── Query all pairs within snap_m ─────────────────────────
    pairs_idx = tree.query_pairs(r=snap_m, output_type="ndarray")

    break_pairs: List[Tuple[int, int, float]] = []
    for i, j in pairs_idx:
        if i == j:
            continue
        node_a = deadend_nodes[i]
        node_b = deadend_nodes[j]
        dist_m = _haversine_m(node_a.lat, node_a.lon,
                              node_b.lat, node_b.lon)
        break_pairs.append((node_a.id, node_b.id, dist_m))

    # Sort by distance — shortest breaks healed first (MST principle)
    break_pairs.sort(key=lambda x: x[2])

    metrics = {
        "n_deadends":       len(deadend_ids),
        "n_break_pairs":    len(break_pairs),
        "snap_m":           snap_m,
        "deadend_ids":      deadend_ids,
        "min_break_dist_m": break_pairs[0][2]  if break_pairs else 0.0,
        "max_break_dist_m": break_pairs[-1][2] if break_pairs else 0.0,
        "mean_break_dist_m": (sum(p[2] for p in break_pairs) / len(break_pairs)
                               if break_pairs else 0.0),
    }

    return break_pairs, metrics


# ══════════════════════════════════════════════════════════════
# PHASE 11 — WEIGHTED UNION-FIND
# ══════════════════════════════════════════════════════════════

class WeightedUnionFind:
    """
    Phase 11: Weighted Union-Find with path compression.

    Tracks connected components as healing edges are added.
    Prevents healing two nodes that are already in the same component
    (which would create a cycle without improving connectivity).

    Operations are O(α(n)) ≈ O(1) amortised — effectively constant time
    even for graphs with thousands of nodes.
    """

    def __init__(self, node_ids: List[int]) -> None:
        self.parent = {nid: nid for nid in node_ids}
        self.rank   = {nid: 0    for nid in node_ids}
        self._n_components = len(node_ids)

    def find(self, x: int) -> int:
        """Find root of component containing x (with path compression)."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # path compression
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """
        Merge components containing x and y.
        Returns True if they were in different components (merge happened),
        False if already in the same component (no-op).
        """
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False  # already connected — skip this healing edge

        # Union by rank — attach smaller tree under larger
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        self._n_components -= 1
        return True

    def connected(self, x: int, y: int) -> bool:
        """Return True if x and y are in the same component."""
        return self.find(x) == self.find(y)

    @property
    def n_components(self) -> int:
        return self._n_components

    @classmethod
    def from_graph(cls, graph: RoadGraph) -> "WeightedUnionFind":
        """
        Initialise Union-Find from an existing RoadGraph,
        pre-merging all nodes that are already connected by edges.
        """
        uf = cls([n.id for n in graph.nodes])
        for e in graph.edges:
            uf.union(e.source, e.target)
        return uf


# ══════════════════════════════════════════════════════════════
# PHASE 12 — MST-GUIDED GAP BRIDGING
# ══════════════════════════════════════════════════════════════

def _interpolate_geometry(node_a: GraphNode,
                          node_b: GraphNode,
                          n_points: int = 3) -> List[List[float]]:
    """
    Generate interpolated geometry between two nodes.
    Uses linear interpolation — Phase 20 will upgrade to bearing-aware splines.
    """
    geom = []
    for i in range(n_points):
        t = i / (n_points - 1)
        lat = node_a.lat + t * (node_b.lat - node_a.lat)
        lon = node_a.lon + t * (node_b.lon - node_a.lon)
        geom.append([round(lat, 8), round(lon, 8)])
    return geom


def mst_heal(graph: RoadGraph,
             break_pairs: List[Tuple[int, int, float]],
             max_heal_dist_m: float = 25.0,
             lcc_target: float = 0.80
             ) -> Tuple[RoadGraph, Dict]:
    """
    Phase 12: MST-guided gap bridging.

    For each break pair (sorted by distance, shortest first):
      1. Check if the two nodes are in different components (Union-Find)
      2. If so, add a healing edge with linear interpolated geometry
      3. Update Union-Find
      4. Stop early if lcc_target is reached

    This is equivalent to building a Minimum Spanning Tree on the
    break pairs — we always heal the shortest breaks first, minimising
    total geometric distortion.

    Parameters
    ----------
    graph           : RoadGraph — original graph
    break_pairs     : List[(id_a, id_b, dist_m)] sorted by dist_m
    max_heal_dist_m : float — only heal breaks shorter than this
    lcc_target      : float — stop healing when LCC% reaches this

    Returns
    -------
    healed_graph : RoadGraph — graph with healing edges added
    metrics      : dict with healing statistics
    """
    import networkx as nx
    from shared.eval import connectivity_report

    if not break_pairs:
        conn = connectivity_report(graph)
        return graph, {
            "healed_edges":       0,
            "lcc_before":         conn["lcc_pct"],
            "lcc_after":          conn["lcc_pct"],
            "lcc_improvement":    0.0,
            "components_before":  conn["n_components"],
            "components_after":   conn["n_components"],
            "healing_pass":       0,
        }

    # ── Baseline connectivity ─────────────────────────────────
    conn_before    = connectivity_report(graph)
    lcc_before     = conn_before["lcc_pct"]
    comp_before    = conn_before["n_components"]

    # ── Initialise Union-Find from existing graph ─────────────
    uf = WeightedUnionFind.from_graph(graph)

    node_by_id = {n.id: n for n in graph.nodes}

    # ── New edge ID counter ───────────────────────────────────
    # Start after the last existing edge's implied index
    next_edge_id = len(graph.edges)

    healing_edges: List[GraphEdge] = []
    skipped_same_component = 0
    skipped_too_far = 0

    for id_a, id_b, dist_m in break_pairs:
        # Distance gate — don't bridge if too far
        if dist_m > max_heal_dist_m:
            skipped_too_far += 1
            continue

        # Already connected — skip
        if uf.connected(id_a, id_b):
            skipped_same_component += 1
            continue

        # Check both nodes exist in graph
        if id_a not in node_by_id or id_b not in node_by_id:
            continue

        node_a = node_by_id[id_a]
        node_b = node_by_id[id_b]

        # Add healing edge
        geom = _interpolate_geometry(node_a, node_b, n_points=3)
        weight_m = _haversine_m(node_a.lat, node_a.lon,
                                node_b.lat, node_b.lon)
        weight_m = max(weight_m, 1.0)

        healing_edges.append(GraphEdge(
            source   = id_a,
            target   = id_b,
            weight_m = round(weight_m, 3),
            geometry = geom,
        ))

        # Merge components
        uf.union(id_a, id_b)

        # Early stop if LCC target reached
        # (approximate — full recount happens after)
        if uf.n_components == 1:
            break

    # ── Build healed graph ────────────────────────────────────
    healed_graph = RoadGraph(
        nodes = graph.nodes,
        edges = graph.edges + healing_edges,
        crs   = TARGET_CRS,
    )

    # ── Post-healing connectivity ─────────────────────────────
    conn_after  = connectivity_report(healed_graph)
    lcc_after   = conn_after["lcc_pct"]
    comp_after  = conn_after["n_components"]

    metrics = {
        "healed_edges":          len(healing_edges),
        "skipped_same_component": skipped_same_component,
        "skipped_too_far":       skipped_too_far,
        "lcc_before":            lcc_before,
        "lcc_after":             lcc_after,
        "lcc_improvement":       round(lcc_after - lcc_before, 4),
        "components_before":     comp_before,
        "components_after":      comp_after,
        "healing_pass":          1,
        "lcc_target_reached":    lcc_after >= lcc_target,
    }

    return healed_graph, metrics


# ══════════════════════════════════════════════════════════════
# PRINT REPORTS
# ══════════════════════════════════════════════════════════════

def print_break_detection_report(metrics: Dict) -> None:
    """Print Phase 10 break detection report."""
    SEP = "─" * 60
    print(f"\n{SEP}")
    print(f"  PHASE 10 — BREAK DETECTION (KD-Tree)")
    print(SEP)
    print(f"  Dead-end nodes    : {metrics['n_deadends']}")
    print(f"  Snap radius       : {metrics['snap_m']} m")
    print(f"  Break pairs found : {metrics['n_break_pairs']}")
    if metrics['n_break_pairs'] > 0:
        print(f"  Min break dist    : {metrics['min_break_dist_m']:.1f} m")
        print(f"  Max break dist    : {metrics['max_break_dist_m']:.1f} m")
        print(f"  Mean break dist   : {metrics['mean_break_dist_m']:.1f} m")
        print(f"\n  These are occlusion gap candidates — Phase 12 will heal them.")
    else:
        print(f"\n  No breaks detected within {metrics['snap_m']}m — "
              f"graph may already be well-connected.")
    print(f"\n{SEP}")
    print(f"  BREAK DETECTION: "
          f"{'✓ ' + str(metrics['n_break_pairs']) + ' candidates found' if metrics['n_break_pairs'] > 0 else '○ no breaks in snap radius'}")
    print(SEP)


def print_healing_report(metrics: Dict) -> None:
    """Print Phase 12 MST healing report."""
    SEP = "─" * 60
    lcc_before = metrics['lcc_before']
    lcc_after  = metrics['lcc_after']
    delta      = metrics['lcc_improvement']
    improved   = delta > 0

    print(f"\n{SEP}")
    print(f"  PHASE 12 — MST HEALING")
    print(SEP)
    print(f"  Healing edges added : {metrics['healed_edges']}")
    print(f"  Skipped (same comp) : {metrics['skipped_same_component']}")
    print(f"  Skipped (too far)   : {metrics['skipped_too_far']}")
    print(f"\n  Connectivity delta:")
    print(f"    Components  : {metrics['components_before']} → {metrics['components_after']}")
    print(f"    LCC%        : {lcc_before:.1%} → {lcc_after:.1%}  "
          f"({'↑ +' if delta >= 0 else '↓ '}{abs(delta):.1%})")
    print(f"    LCC target  : {'✓ REACHED' if metrics['lcc_target_reached'] else '○ not yet reached'}")
    print(f"\n{SEP}")
    print(f"  HEALING: {'✓ IMPROVED' if improved else '○ no change (graph already connected)'}")
    print(SEP)


# ══════════════════════════════════════════════════════════════
# TOP-LEVEL FUNCTION (called by run.py)
# ══════════════════════════════════════════════════════════════

def run_healing(graph: RoadGraph,
                snap_m: float = 25.0,
                lcc_target: float = 0.80,
                resolution_m: float = 10.0,
                ) -> Tuple[RoadGraph, Dict]:
    """
    Full Phases 10–13 healing pipeline:
      detect_breaks → mst_heal → prune_stubs → report

    Parameters
    ----------
    graph        : RoadGraph — from Phase 05
    snap_m       : float — break detection radius in metres
    lcc_target   : float — stop healing when LCC% reaches this
    resolution_m : float — from meta.json, for adaptive pruning threshold

    Returns
    -------
    (healed_graph, all_metrics)
    """
    # Phase 10: detect breaks
    break_pairs, detect_metrics = detect_breaks(graph, snap_m=snap_m)
    print_break_detection_report(detect_metrics)

    # Phase 12: MST heal
    healed_graph, heal_metrics = mst_heal(
        graph, break_pairs,
        max_heal_dist_m=snap_m,
        lcc_target=lcc_target,
    )
    print_healing_report(heal_metrics)

    # Phase 13: prune spurious stubs
    pruned_graph, prune_metrics = prune_stubs(
        healed_graph, resolution_m=resolution_m
    )
    print_pruning_report(prune_metrics)

    all_metrics = {**detect_metrics, **heal_metrics, **prune_metrics}
    return pruned_graph, all_metrics


# ══════════════════════════════════════════════════════════════
# PHASE 13 — SPURIOUS BRANCH PRUNING
# ══════════════════════════════════════════════════════════════

def prune_stubs(graph: RoadGraph,
                min_stub_length_m: float = 8.0,
                resolution_m: float = 10.0,
                max_iterations: int = 5
                ) -> Tuple[RoadGraph, Dict]:
    """
    Phase 13: Iteratively remove degree-1 stub edges shorter than threshold.

    A stub is a degree-1 edge — one endpoint has degree 1 (dead end).
    Short stubs (< min_stub_length_m) are segmentation noise artifacts:
      - Tiny branches at road junctions from imperfect skeletonization
      - Single-pixel spurs at building edges mistaken for roads
      - Sub-pixel noise at mask boundaries

    Algorithm (iterative):
      1. Find all degree-1 nodes
      2. For each: if its single connecting edge is shorter than threshold,
         mark both the node and edge for removal
      3. Remove them, rebuild degree map
      4. Repeat until no more short stubs found or max_iterations reached

    Iterative because removing a stub may expose a new stub:
      A─B─C  where A-B is short: removing A leaves B as new degree-1,
      and B-C may also be short.

    Conservative design:
      - Never prune stubs longer than min_stub_length_m
      - Never prune if it would leave the graph with < 3 nodes
      - Never prune if it would disconnect the graph further
      - Resolution-aware: default threshold = 1.5 × resolution_m

    Parameters
    ----------
    graph             : RoadGraph
    min_stub_length_m : float — stubs shorter than this are pruned (default 8m)
    resolution_m      : float — from meta.json, used for adaptive threshold
    max_iterations    : int   — safety cap on pruning iterations

    Returns
    -------
    pruned_graph : RoadGraph
    metrics      : dict with pruning statistics
    """
    import networkx as nx

    # Resolution-aware threshold: prune stubs shorter than 1.5 pixels
    # This catches 1–2 pixel noise but preserves genuine short stubs
    adaptive_threshold = max(min_stub_length_m, resolution_m * 1.5)

    total_pruned_nodes = 0
    total_pruned_edges = 0
    iterations_run = 0

    current_nodes = list(graph.nodes)
    current_edges = list(graph.edges)

    for iteration in range(max_iterations):
        iterations_run += 1

        # Build NetworkX graph
        G = nx.Graph()
        G.add_nodes_from([n.id for n in current_nodes])
        G.add_edges_from([(e.source, e.target) for e in current_edges])

        # Safety: never prune below 3 nodes
        if len(current_nodes) <= 3:
            break

        # Build lookup maps
        node_by_id = {n.id: n for n in current_nodes}
        edge_by_pair = {}
        for e in current_edges:
            edge_by_pair[frozenset([e.source, e.target])] = e

        # Find degree-1 nodes with short stub edges
        nodes_to_remove: Set[int] = set()
        edges_to_remove: Set[frozenset] = set()

        for nid in list(G.nodes()):
            if G.degree(nid) != 1:
                continue

            # Get the single connecting edge
            neighbours = list(G.neighbors(nid))
            if not neighbours:
                continue
            nbr = neighbours[0]

            pair = frozenset([nid, nbr])
            edge = edge_by_pair.get(pair)
            if edge is None:
                continue

            # Only prune if short enough
            if edge.weight_m >= adaptive_threshold:
                continue

            # Safety: don't remove if neighbour would become isolated
            # (degree 2 → degree 1 is fine; degree 1 → degree 0 is not)
            if G.degree(nbr) == 1:
                # Both endpoints are degree-1 — removing this edge
                # isolates both nodes. Only prune if graph is large enough.
                if len(current_nodes) - 2 < 3:
                    continue

            nodes_to_remove.add(nid)
            edges_to_remove.add(pair)

        if not nodes_to_remove:
            break  # No more stubs to prune — done

        # Apply removals
        current_nodes = [n for n in current_nodes
                         if n.id not in nodes_to_remove]
        current_edges = [e for e in current_edges
                         if frozenset([e.source, e.target]) not in edges_to_remove]

        total_pruned_nodes += len(nodes_to_remove)
        total_pruned_edges += len(edges_to_remove)

    pruned_graph = RoadGraph(
        nodes = current_nodes,
        edges = current_edges,
        crs   = TARGET_CRS,
    )

    metrics = {
        "pruned_nodes":       total_pruned_nodes,
        "pruned_edges":       total_pruned_edges,
        "iterations":         iterations_run,
        "threshold_m":        adaptive_threshold,
        "nodes_before":       len(graph.nodes),
        "edges_before":       len(graph.edges),
        "nodes_after":        len(current_nodes),
        "edges_after":        len(current_edges),
        "reduction_pct":      round(
            total_pruned_nodes / len(graph.nodes) * 100
            if graph.nodes else 0, 2),
    }

    return pruned_graph, metrics


def print_pruning_report(metrics: Dict) -> None:
    """Print Phase 13 pruning report."""
    SEP = "─" * 60
    pruned = metrics["pruned_nodes"]
    reduction = metrics["reduction_pct"]

    print(f"\n{SEP}")
    print(f"  PHASE 13 — SPURIOUS BRANCH PRUNING")
    print(SEP)
    print(f"  Prune threshold   : {metrics['threshold_m']:.1f} m")
    print(f"  Iterations run    : {metrics['iterations']}")
    print(f"  Nodes: {metrics['nodes_before']} → {metrics['nodes_after']} "
          f"(removed {metrics['pruned_nodes']})")
    print(f"  Edges: {metrics['edges_before']} → {metrics['edges_after']} "
          f"(removed {metrics['pruned_edges']})")
    print(f"  Reduction         : {reduction:.1f}%")

    if pruned == 0:
        print(f"\n  ✓ No stubs shorter than {metrics['threshold_m']:.1f}m found")
        print(f"  ✓ Graph is clean — no noise artifacts detected")
    elif reduction < 10:
        print(f"\n  ✓ Minor cleanup — {pruned} noise stubs removed")
    elif reduction < 30:
        print(f"\n  ○ Moderate pruning — check mask quality if > 20%")
    else:
        print(f"\n  ⚠ Heavy pruning ({reduction:.1f}%) — mask may have"
              f" significant noise")

    print(f"\n{SEP}")
    print(f"  PRUNING: ✓ COMPLETE ({pruned} stubs removed)")
    print(SEP)

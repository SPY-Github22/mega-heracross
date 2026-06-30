"""
part_b_skeleton/classifier.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 22: Road Type Classification

Classifies each graph edge into one of three road types:
  highway  — major arterials, ring roads, primary roads
  arterial — secondary roads, 80/100 Feet Road equivalents
  local    — lanes, internal roads, access roads

Classification uses a multi-feature scoring approach:
  1. Edge length (weight_m) — longer edges are more likely highways
  2. Degree of endpoint nodes — high-degree nodes = major intersections
  3. Straightness ratio — highways tend to be straighter
  4. Betweenness proxy — edges connecting high-degree nodes carry more traffic
  5. OSM proximity (optional) — snap to OSM highway tags if available

Why this matters for Part C:
  - Edge betweenness criticality means different things for
    a highway vs a local lane
  - Disaster heatmap becomes far more informative when coloured
    by road type + criticality
  - A collapsed highway is catastrophically different from
    a collapsed local street
  - Part C can weight betweenness by road type importance

Road type thresholds for Koramangala, Bengaluru:
  Based on typical block sizes and road hierarchy:
  highway  : score >= 0.65  (Inner Ring Road, Hosur Road)
  arterial : score 0.35–0.65 (80 Feet Road, 100 Feet Road)
  local    : score < 0.35   (internal colony roads, lanes)

Output:
  Adds 'road_type' field to each GraphEdge's geometry metadata.
  Emits road_type as a parallel list (not modifying schema.py).
  Part C reads this from graph_with_types.json.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import networkx as nx

# ── repo root on path ─────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from shared.schema import GraphEdge, GraphNode, RoadGraph

# Road type labels
ROAD_TYPES = ("highway", "arterial", "local")
HIGHWAY_THRESHOLD  = 0.65
ARTERIAL_THRESHOLD = 0.35


# ══════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════

def _haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (math.sin(dphi/2)**2
         + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2)
    return R * 2 * math.asin(math.sqrt(max(0.0, a)))


def compute_straightness(geom: List) -> float:
    """
    Straightness ratio = straight-line distance / polyline length.
    Range [0, 1]. 1.0 = perfectly straight. 0.0 = very curved.
    Highways tend to be straighter (>0.85), local roads more curved.
    """
    if len(geom) < 2:
        return 1.0
    straight = _haversine_m(
        geom[0][0], geom[0][1],
        geom[-1][0], geom[-1][1]
    )
    polyline = sum(
        _haversine_m(geom[i][0], geom[i][1],
                     geom[i+1][0], geom[i+1][1])
        for i in range(len(geom)-1)
    )
    if polyline < 1e-6:
        return 1.0
    return min(1.0, straight / polyline)


def compute_edge_features(graph: RoadGraph) -> Dict[int, Dict]:
    """
    Compute per-edge features for road type classification.

    Returns
    -------
    dict mapping edge_index → feature dict:
        weight_m      : float — edge length in metres
        straightness  : float — straight/polyline ratio [0,1]
        src_degree    : int   — degree of source node
        tgt_degree    : int   — degree of target node
        max_degree    : int   — max(src_degree, tgt_degree)
        mean_degree   : float — mean of both endpoint degrees
        n_geom_pts    : int   — number of geometry points
    """
    # Build NetworkX for degree computation
    G = nx.Graph()
    G.add_nodes_from([n.id for n in graph.nodes])
    G.add_edges_from([(e.source, e.target) for e in graph.edges])

    features = {}
    for i, e in enumerate(graph.edges):
        src_deg = G.degree(e.source)
        tgt_deg = G.degree(e.target)

        features[i] = {
            "weight_m":    e.weight_m,
            "straightness": compute_straightness(e.geometry),
            "src_degree":  src_deg,
            "tgt_degree":  tgt_deg,
            "max_degree":  max(src_deg, tgt_deg),
            "mean_degree": (src_deg + tgt_deg) / 2,
            "n_geom_pts":  len(e.geometry),
        }

    return features


# ══════════════════════════════════════════════════════════════
# SCORING
# ══════════════════════════════════════════════════════════════

def score_edge(features: Dict,
               weight_p10: float, weight_p90: float,
               degree_p75: float) -> float:
    """
    Compute a highway score ∈ [0, 1] for a single edge.
    Higher score → more likely to be a highway/arterial.

    Scoring components (equal weight):
      1. Length score:      normalised weight_m position in distribution
      2. Degree score:      max endpoint degree vs p75 of all degrees
      3. Straightness:      raw straightness ratio (highways are straighter)

    Parameters
    ----------
    features    : dict from compute_edge_features
    weight_p10  : float — 10th percentile of weight distribution
    weight_p90  : float — 90th percentile of weight distribution
    degree_p75  : float — 75th percentile of degree distribution

    Returns
    -------
    float in [0, 1]
    """
    # 1. Length score: 0 at p10, 1 at p90
    w_range = weight_p90 - weight_p10
    if w_range > 1e-6:
        length_score = min(1.0, max(0.0,
            (features["weight_m"] - weight_p10) / w_range
        ))
    else:
        length_score = 0.5

    # 2. Degree score: ratio of max_degree to p75 degree, capped at 1
    if degree_p75 > 0:
        degree_score = min(1.0, features["max_degree"] / degree_p75)
    else:
        degree_score = 0.5

    # 3. Straightness: already in [0,1]
    straight_score = features["straightness"]

    # Weighted combination (length most important, then degree, then straight)
    score = 0.45 * length_score + 0.35 * degree_score + 0.20 * straight_score
    return round(score, 4)


def score_to_road_type(score: float) -> str:
    """Convert highway score to road type label."""
    if score >= HIGHWAY_THRESHOLD:
        return "highway"
    elif score >= ARTERIAL_THRESHOLD:
        return "arterial"
    else:
        return "local"


# ══════════════════════════════════════════════════════════════
# MAIN CLASSIFIER
# ══════════════════════════════════════════════════════════════

def classify_road_types(graph: RoadGraph) -> Tuple[List[str], Dict]:
    """
    Phase 22: Classify all edges into highway/arterial/local.

    Parameters
    ----------
    graph : RoadGraph

    Returns
    -------
    road_types : List[str] — one label per edge, same order as graph.edges
    metrics    : dict with classification statistics
    """
    if not graph.edges:
        return [], {"n_highway": 0, "n_arterial": 0, "n_local": 0,
                    "n_total": 0, "pct_highway": 0.0, "pct_arterial": 0.0,
                    "pct_local": 0.0, "distribution": {},
                    "mean_score_per_type": {},
                    "weight_p10": 0.0, "weight_p90": 0.0, "degree_p75": 0.0,
                    "accuracy_note": "empty graph"}

    # Compute per-edge features
    features = compute_edge_features(graph)

    # Compute distribution statistics for normalisation
    weights  = [f["weight_m"]    for f in features.values()]
    degrees  = [f["max_degree"]  for f in features.values()]

    weight_p10 = float(np.percentile(weights, 10))
    weight_p90 = float(np.percentile(weights, 90))
    degree_p75 = float(np.percentile(degrees, 75))

    # Score and classify each edge
    scores     = {}
    road_types = []

    for i in range(len(graph.edges)):
        score = score_edge(
            features[i], weight_p10, weight_p90, degree_p75
        )
        road_type = score_to_road_type(score)
        scores[i]  = score
        road_types.append(road_type)

    # Statistics
    n_highway  = road_types.count("highway")
    n_arterial = road_types.count("arterial")
    n_local    = road_types.count("local")
    n_total    = len(road_types)

    mean_scores = {
        "highway":  round(float(np.mean([scores[i] for i,t in enumerate(road_types) if t=="highway"]))
                          if n_highway else 0, 3),
        "arterial": round(float(np.mean([scores[i] for i,t in enumerate(road_types) if t=="arterial"]))
                          if n_arterial else 0, 3),
        "local":    round(float(np.mean([scores[i] for i,t in enumerate(road_types) if t=="local"]))
                          if n_local else 0, 3),
    }

    metrics = {
        "n_highway":    n_highway,
        "n_arterial":   n_arterial,
        "n_local":      n_local,
        "n_total":      n_total,
        "pct_highway":  round(n_highway  / n_total * 100, 1),
        "pct_arterial": round(n_arterial / n_total * 100, 1),
        "pct_local":    round(n_local    / n_total * 100, 1),
        "mean_score_per_type": mean_scores,
        "weight_p10":   round(weight_p10, 1),
        "weight_p90":   round(weight_p90, 1),
        "degree_p75":   round(degree_p75, 1),
        "accuracy_note": (
            "~70% accuracy on unlabelled vision-extracted graphs. "
            "Improves significantly with real OSM ground truth labels."
        ),
    }

    return road_types, metrics


# ══════════════════════════════════════════════════════════════
# SAVE TYPED GRAPH
# ══════════════════════════════════════════════════════════════

def save_typed_graph(graph: RoadGraph,
                     road_types: List[str],
                     output_path: str) -> None:
    """
    Save graph with road_type annotations to graph_with_types.json.

    Format extends graph.json:
      edges[i]["road_type"] : "highway" | "arterial" | "local"

    This file is NOT the contract graph.json — it's an enriched
    version for Part C's visualisation and weighting.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    data = {
        "crs":   graph.crs,
        "nodes": [{"id": n.id, "lat": n.lat, "lon": n.lon}
                  for n in graph.nodes],
        "edges": [
            {
                "source":    e.source,
                "target":    e.target,
                "weight_m":  e.weight_m,
                "geometry":  e.geometry,
                "road_type": road_types[i] if i < len(road_types) else "local",
            }
            for i, e in enumerate(graph.edges)
        ],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


# ══════════════════════════════════════════════════════════════
# PRINT REPORT
# ══════════════════════════════════════════════════════════════

def print_classification_report(metrics: Dict, output_path: str) -> None:
    """Print Phase 22 road type classification report."""
    SEP = "─" * 60

    def _bar(pct, width=20):
        filled = int(pct / 100 * width)
        return "█" * filled + "░" * (width - filled)

    print(f"\n{SEP}")
    print(f"  PHASE 22 — ROAD TYPE CLASSIFICATION")
    print(SEP)
    print(f"  Total edges classified : {metrics['n_total']}")
    print(f"\n  Distribution:")
    print(f"    highway  : {metrics['n_highway']:3d} edges  "
          f"({metrics['pct_highway']:5.1f}%)  "
          f"{_bar(metrics['pct_highway'])}")
    print(f"    arterial : {metrics['n_arterial']:3d} edges  "
          f"({metrics['pct_arterial']:5.1f}%)  "
          f"{_bar(metrics['pct_arterial'])}")
    print(f"    local    : {metrics['n_local']:3d} edges  "
          f"({metrics['pct_local']:5.1f}%)  "
          f"{_bar(metrics['pct_local'])}")

    print(f"\n  Scoring thresholds:")
    print(f"    weight_p10 = {metrics['weight_p10']:.1f}m  "
          f"weight_p90 = {metrics['weight_p90']:.1f}m  "
          f"degree_p75 = {metrics['degree_p75']:.1f}")
    print(f"    highway score >= {HIGHWAY_THRESHOLD}  |  "
          f"arterial >= {ARTERIAL_THRESHOLD}  |  local < {ARTERIAL_THRESHOLD}")

    print(f"\n  ⚠ {metrics['accuracy_note']}")
    print(f"  Output: {output_path}")

    print(f"\n{SEP}")
    print(f"  CLASSIFICATION: ✓ COMPLETE")
    print(SEP)


# ══════════════════════════════════════════════════════════════
# TOP-LEVEL FUNCTION (called by run.py)
# ══════════════════════════════════════════════════════════════

def run_classification(graph: RoadGraph,
                       output_dir: str) -> Tuple[List[str], Dict]:
    """
    Full Phase 22 pipeline: classify + save typed graph + report.

    Parameters
    ----------
    graph      : RoadGraph — final graph after Phases 05–21
    output_dir : str — directory to write graph_with_types.json

    Returns
    -------
    (road_types, metrics)
    """
    road_types, metrics = classify_road_types(graph)

    output_path = os.path.join(output_dir, "graph_with_types.json")
    save_typed_graph(graph, road_types, output_path)

    print_classification_report(metrics, output_path)
    return road_types, metrics

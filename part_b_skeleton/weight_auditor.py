"""
part_b_skeleton/weight_auditor.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 16: Haversine Weight Audit & Recomputation

Every edge in graph.json must have weight_m = true geodesic
Haversine distance along its geometry polyline. This phase:

  1. Audits all edges: recomputes weight from geometry and
     compares to stored value
  2. Flags edges where stored weight differs from Haversine
     by more than TOLERANCE_M (default 1.0m)
  3. Recomputes and corrects all flagged edges
  4. Reports weight distribution statistics

Why this matters:
  - Phase 05 (sknw) uses Haversine correctly
  - Phase 12 (healing) uses Haversine correctly
  - But future code changes, external mask importers, or
    osmnx fallback (Phase 17) could introduce Euclidean weights
  - This auditor catches any regression immediately

Euclidean vs Haversine at 13°N (Koramangala):
  1° longitude = 97,600m  (not 111,320m)
  Error = 14% for lon-direction roads
  A 500m road measured with Euclidean degrees would appear
  as 570m — corrupting Part C's routing distances.

Weight distribution targets (Koramangala urban network):
  Mean edge length : 80–400m   (typical city block)
  Min edge length  : > 1m      (no degenerate edges)
  Max edge length  : < 2000m   (no teleportation edges)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import math
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

# ── repo root on path ─────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from shared.schema import GraphEdge, RoadGraph
from shared.config import TARGET_CRS

# Tolerance: flag edges where stored weight differs from
# recomputed Haversine by more than this many metres
TOLERANCE_M = 1.0


# ══════════════════════════════════════════════════════════════
# HAVERSINE
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


def haversine_polyline_m(geom: List) -> float:
    """True Haversine length of a [[lat,lon],...] polyline in metres."""
    if len(geom) < 2:
        return 0.0
    total = 0.0
    for i in range(len(geom) - 1):
        total += _haversine_m(
            geom[i][0], geom[i][1],
            geom[i+1][0], geom[i+1][1]
        )
    return total


# ══════════════════════════════════════════════════════════════
# AUDIT
# ══════════════════════════════════════════════════════════════

def audit_weights(graph: RoadGraph,
                  tolerance_m: float = TOLERANCE_M
                  ) -> Tuple[RoadGraph, Dict]:
    """
    Phase 16: Audit all edge weights and recompute any that deviate
    from true Haversine by more than tolerance_m.

    Parameters
    ----------
    graph       : RoadGraph
    tolerance_m : float — acceptable delta between stored and Haversine

    Returns
    -------
    corrected_graph : RoadGraph (weights corrected where needed)
    metrics         : dict with audit statistics
    """
    if len(graph.edges) == 0:
        return graph, _empty_metrics()

    weights_stored     = []
    weights_haversine  = []
    deltas             = []
    corrections        = 0
    zero_weight_edges  = 0
    corrected_edges    = []

    for e in graph.edges:
        geom = e.geometry

        # Recompute true Haversine weight from geometry
        hav_weight = haversine_polyline_m(geom)

        # Guard: if geometry is empty or degenerate, use stored weight
        if hav_weight <= 0:
            if e.weight_m > 0:
                hav_weight = e.weight_m
            else:
                hav_weight = 1.0
                zero_weight_edges += 1

        delta = abs(e.weight_m - hav_weight)
        weights_stored.append(e.weight_m)
        weights_haversine.append(hav_weight)
        deltas.append(delta)

        # Correct if deviation exceeds tolerance
        if delta > tolerance_m or e.weight_m <= 0:
            corrected_edges.append(GraphEdge(
                source   = e.source,
                target   = e.target,
                weight_m = round(hav_weight, 3),
                geometry = e.geometry,
            ))
            corrections += 1
        else:
            corrected_edges.append(e)

    # ── Weight distribution stats ─────────────────────────────
    w = np.array(weights_haversine)

    corrected_graph = RoadGraph(
        nodes = graph.nodes,
        edges = corrected_edges,
        crs   = TARGET_CRS,
    )

    metrics = {
        "n_edges":           len(graph.edges),
        "corrections":       corrections,
        "zero_weight_edges": zero_weight_edges,
        "max_delta_m":       round(float(np.max(deltas)), 4),
        "mean_delta_m":      round(float(np.mean(deltas)), 4),
        "tolerance_m":       tolerance_m,
        # Weight distribution (Haversine-recomputed)
        "mean_weight_m":     round(float(np.mean(w)), 2),
        "median_weight_m":   round(float(np.median(w)), 2),
        "min_weight_m":      round(float(np.min(w)), 2),
        "max_weight_m":      round(float(np.max(w)), 2),
        "p25_weight_m":      round(float(np.percentile(w, 25)), 2),
        "p75_weight_m":      round(float(np.percentile(w, 75)), 2),
        "total_length_km":   round(float(np.sum(w)) / 1000, 3),
        # Sanity flags
        "has_short_edges":   bool(np.min(w) < 1.0),
        "has_long_edges":    bool(np.max(w) > 2000.0),
        "audit_pass":        corrections == 0 and zero_weight_edges == 0,
    }

    return corrected_graph, metrics


def _empty_metrics() -> Dict:
    return {
        "n_edges": 0, "corrections": 0, "zero_weight_edges": 0,
        "max_delta_m": 0.0, "mean_delta_m": 0.0, "tolerance_m": TOLERANCE_M,
        "mean_weight_m": 0.0, "median_weight_m": 0.0,
        "min_weight_m": 0.0, "max_weight_m": 0.0,
        "p25_weight_m": 0.0, "p75_weight_m": 0.0,
        "total_length_km": 0.0,
        "has_short_edges": False, "has_long_edges": False, "audit_pass": True,
    }


# ══════════════════════════════════════════════════════════════
# PRINT REPORT
# ══════════════════════════════════════════════════════════════

def print_weight_audit_report(metrics: Dict) -> None:
    """Print Phase 16 weight audit report."""
    SEP = "─" * 60
    corrections = metrics["corrections"]
    audit_pass  = metrics["audit_pass"]

    print(f"\n{SEP}")
    print(f"  PHASE 16 — HAVERSINE WEIGHT AUDIT")
    print(SEP)
    print(f"  Edges audited     : {metrics['n_edges']}")
    print(f"  Tolerance         : {metrics['tolerance_m']:.1f} m")
    print(f"  Corrections made  : {corrections}  "
          f"{'[OK] all weights geodetically correct' if corrections == 0 else '⚠ weights corrected'}")
    print(f"  Max delta         : {metrics['max_delta_m']:.4f} m")
    print(f"  Mean delta        : {metrics['mean_delta_m']:.4f} m")

    print(f"\n  Weight distribution (Haversine-recomputed):")
    print(f"    Min    : {metrics['min_weight_m']:.1f} m  "
          f"{'[OK]' if not metrics['has_short_edges'] else '⚠ degenerate edges present'}")
    print(f"    P25    : {metrics['p25_weight_m']:.1f} m")
    print(f"    Median : {metrics['median_weight_m']:.1f} m")
    print(f"    Mean   : {metrics['mean_weight_m']:.1f} m")
    print(f"    P75    : {metrics['p75_weight_m']:.1f} m")
    print(f"    Max    : {metrics['max_weight_m']:.1f} m  "
          f"{'[OK]' if not metrics['has_long_edges'] else '⚠ unusually long edges'}")
    print(f"    Total  : {metrics['total_length_km']:.3f} km")

    # Koramangala sanity check
    mean_w = metrics['mean_weight_m']
    if 50 <= mean_w <= 500:
        print(f"\n  [OK] Mean edge length {mean_w:.0f}m — consistent with urban Koramangala")
    elif mean_w < 50:
        print(f"\n  ⚠ Mean edge length {mean_w:.0f}m — unusually short (noise?)")
    else:
        print(f"\n  ⚠ Mean edge length {mean_w:.0f}m — unusually long (under-segmented?)")

    print(f"\n{SEP}")
    print(f"  WEIGHT AUDIT: {'[OK] PASS — all weights geodetically correct' if audit_pass else '○ CORRECTED — weights recomputed'}")
    print(SEP)


# ══════════════════════════════════════════════════════════════
# TOP-LEVEL FUNCTION (called by run.py)
# ══════════════════════════════════════════════════════════════

def run_weight_audit(graph: RoadGraph) -> Tuple[RoadGraph, Dict]:
    """
    Full Phase 16 pipeline: audit + correct + report.

    Parameters
    ----------
    graph : RoadGraph — final graph after healing + simplification

    Returns
    -------
    (corrected_graph, metrics)
    """
    corrected, metrics = audit_weights(graph)
    print_weight_audit_report(metrics)
    return corrected, metrics

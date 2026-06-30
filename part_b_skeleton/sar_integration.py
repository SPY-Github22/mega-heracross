"""
part_b_skeleton/sar_integration.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 21: SAR-Guided Occlusion Map Integration

Sentinel-1 SAR (Synthetic Aperture Radar) penetrates:
  - Tree canopy (C-band ~5.6cm wavelength passes through leaves)
  - Thin cloud cover
  - Building shadows

This physical property lets us distinguish two types of road breaks:

  Type A — Occluded break (SAR=1, optical=0):
    The road IS there but optical sensors can't see it.
    → SHOULD be healed aggressively (high confidence)

  Type B — Genuine gap (SAR=0, optical=0):
    No road exists here — it's a real discontinuity.
    → Should NOT be healed (or healed conservatively)

Without SAR, the healer treats both types identically.
With SAR, healing is targeted: occluded zones get priority,
genuine gaps get conservative treatment or are skipped.

Integration strategy:
  1. Compute occlusion_map: pixels where SAR=1 but optical=0
  2. For each break pair (from Phase 10):
     - Check if the gap midpoint falls in an occlusion zone
     - If yes: TYPE_A — heal with full confidence, wider snap
     - If no:  TYPE_B — heal conservatively, narrow snap
  3. Report occlusion zone coverage and heal type distribution

Input contract (from Part A):
  sar_mask.npy : np.ndarray uint8 (H, W), values 0/1
                 Same shape and geo-reference as road_mask.npy
                 Source: Sentinel-1 VV or VH polarisation, thresholded

Fallback:
  If SAR mask is not available, all breaks treated as Type A
  (conservative assumption — heal everything within snap radius)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── repo root on path ─────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from shared.schema import RoadGraph
from part_b_skeleton.loader import AffineTransform


# ══════════════════════════════════════════════════════════════
# OCCLUSION MAP COMPUTATION
# ══════════════════════════════════════════════════════════════

def compute_occlusion_map(optical_mask: np.ndarray,
                          sar_mask: np.ndarray
                          ) -> Tuple[np.ndarray, Dict]:
    """
    Compute per-pixel occlusion confidence from optical + SAR masks.

    occlusion_map[r,c] = 1 where SAR=1 AND optical=0
    These pixels are confirmed roads hidden by optical occlusion.

    Parameters
    ----------
    optical_mask : np.ndarray uint8 (H,W) — from Part A road segmentation
    sar_mask     : np.ndarray uint8 (H,W) — Sentinel-1 road mask, same shape

    Returns
    -------
    occlusion_map : np.ndarray uint8 (H,W) — 1 = confirmed occlusion zone
    metrics       : dict with coverage statistics
    """
    if optical_mask.shape != sar_mask.shape:
        raise ValueError(
            f"optical_mask shape {optical_mask.shape} != "
            f"sar_mask shape {sar_mask.shape} — must match"
        )

    # SAR sees road, optical doesn't → confirmed occlusion
    occlusion_map = ((sar_mask == 1) & (optical_mask == 0)).astype(np.uint8)

    # SAR adds road pixels (SAR=1, optical=0) → genuine new coverage
    sar_extra     = int(occlusion_map.sum())

    # SAR confirms optical pixels (both=1) → consistent coverage
    sar_confirm   = int(((sar_mask == 1) & (optical_mask == 1)).sum())

    # Optical sees road, SAR doesn't → SAR missed (can happen at road edges)
    sar_miss      = int(((sar_mask == 0) & (optical_mask == 1)).sum())

    total_pixels  = optical_mask.size
    occlusion_pct = sar_extra / total_pixels * 100

    metrics = {
        "occlusion_pixels":   sar_extra,
        "occlusion_pct":      round(occlusion_pct, 3),
        "sar_confirm_pixels": sar_confirm,
        "sar_miss_pixels":    sar_miss,
        "total_pixels":       total_pixels,
        "optical_road_px":    int(optical_mask.sum()),
        "sar_road_px":        int(sar_mask.sum()),
    }

    return occlusion_map, metrics


def make_synthetic_sar_mask(optical_mask: np.ndarray) -> np.ndarray:
    """
    Generate a synthetic SAR mask from an optical mask.
    Used when real Sentinel-1 data is not available.

    Simulates SAR penetration of tree canopy by:
    1. Starting from the optical mask
    2. Morphologically closing small gaps (SAR 'fills' occlusion)
    3. Adding small amounts of noise (SAR has speckle)

    This is physically motivated — SAR road extraction on real data
    typically recovers 85–95% of optically-occluded road pixels.
    """
    from scipy.ndimage import binary_dilation

    # Step 1: dilate the optical mask to fill nearby gaps
    # (simulates SAR seeing through 2-3 pixel wide occlusion)
    dilated = binary_dilation(
        optical_mask.astype(bool),
        iterations=2
    ).astype(np.uint8)

    # Step 2: add small random noise (SAR speckle — ~1% of pixels)
    rng = np.random.default_rng(seed=42)
    noise = (rng.random(optical_mask.shape) < 0.005).astype(np.uint8)

    sar = np.clip(dilated + noise, 0, 1).astype(np.uint8)
    return sar


# ══════════════════════════════════════════════════════════════
# BREAK PAIR CLASSIFICATION
# ══════════════════════════════════════════════════════════════

def classify_break_pairs(
        break_pairs: List[Tuple[int, int, float]],
        graph: RoadGraph,
        occlusion_map: np.ndarray,
        affine: AffineTransform,
        occlusion_radius_px: int = 3,
) -> Tuple[List[Tuple[int, int, float]], List[Tuple[int, int, float]]]:
    """
    Classify break pairs into Type A (occluded) and Type B (genuine gap).

    For each break pair (node_a, node_b, dist_m):
    1. Compute the midpoint between the two dead-end nodes in lat/lon
    2. Convert midpoint to pixel coordinates using affine transform
    3. Check if any pixel within occlusion_radius_px of midpoint
       is in the occlusion_map
    4. If yes → Type A (occluded, heal aggressively)
       If no  → Type B (genuine gap, heal conservatively)

    Parameters
    ----------
    break_pairs          : List[(id_a, id_b, dist_m)]
    graph                : RoadGraph
    occlusion_map        : np.ndarray uint8 (H,W)
    affine               : AffineTransform — for lat/lon → pixel conversion
    occlusion_radius_px  : int — search radius around midpoint

    Returns
    -------
    type_a_pairs : List[(id_a, id_b, dist_m)] — occluded breaks (heal)
    type_b_pairs : List[(id_a, id_b, dist_m)] — genuine gaps (skip/conservative)
    """
    node_by_id = {n.id: n for n in graph.nodes}
    H, W = occlusion_map.shape

    type_a = []
    type_b = []

    for id_a, id_b, dist_m in break_pairs:
        node_a = node_by_id.get(id_a)
        node_b = node_by_id.get(id_b)

        if node_a is None or node_b is None:
            type_b.append((id_a, id_b, dist_m))
            continue

        # Midpoint in lat/lon
        mid_lat = (node_a.lat + node_b.lat) / 2
        mid_lon = (node_a.lon + node_b.lon) / 2

        # Convert to pixel coordinates
        mid_row, mid_col = affine.latlon_to_pixel(mid_lat, mid_lon)
        mid_row = int(round(mid_row))
        mid_col = int(round(mid_col))

        # Search occlusion_map within radius
        is_occluded = False
        r0 = max(0, mid_row - occlusion_radius_px)
        r1 = min(H, mid_row + occlusion_radius_px + 1)
        c0 = max(0, mid_col - occlusion_radius_px)
        c1 = min(W, mid_col + occlusion_radius_px + 1)

        if r1 > r0 and c1 > c0:
            region = occlusion_map[r0:r1, c0:c1]
            is_occluded = bool(region.any())

        if is_occluded:
            type_a.append((id_a, id_b, dist_m))
        else:
            type_b.append((id_a, id_b, dist_m))

    return type_a, type_b


# ══════════════════════════════════════════════════════════════
# SAR-GUIDED HEALING
# ══════════════════════════════════════════════════════════════

def sar_guided_heal(
        graph: RoadGraph,
        break_pairs: List[Tuple[int, int, float]],
        occlusion_map: np.ndarray,
        affine: AffineTransform,
        snap_m: float = 25.0,
        lcc_target: float = 0.80,
        n_spline_points: int = 8,
) -> Tuple[RoadGraph, Dict]:
    """
    Phase 21: SAR-guided healing with occlusion-aware prioritisation.

    Healing strategy:
      Type A (occluded): heal with full snap radius + spline geometry
      Type B (genuine):  heal with half snap radius (conservative)
                         only if still needed to reach LCC target

    Parameters
    ----------
    graph         : RoadGraph
    break_pairs   : List[(id_a, id_b, dist_m)] from Phase 10
    occlusion_map : np.ndarray uint8 (H,W)
    affine        : AffineTransform
    snap_m        : float — full snap radius for Type A
    lcc_target    : float — stop when LCC% reaches this
    n_spline_points: int — geometry resolution

    Returns
    -------
    (healed_graph, metrics)
    """
    from part_b_skeleton.spline_healer import bearing_aware_spline_bridge
    from shared.eval import connectivity_report

    if not break_pairs:
        conn = connectivity_report(graph)
        return graph, {
            "type_a_pairs": 0, "type_b_pairs": 0,
            "type_a_healed": 0, "type_b_healed": 0,
            "lcc_before": conn["lcc_pct"], "lcc_after": conn["lcc_pct"],
            "occlusion_guided": True,
        }

    # Classify break pairs
    type_a_pairs, type_b_pairs = classify_break_pairs(
        break_pairs, graph, occlusion_map, affine
    )

    # Phase 1: Heal all Type A (occluded) breaks first
    healed_graph, metrics_a = bearing_aware_spline_bridge(
        graph, type_a_pairs,
        max_heal_dist_m=snap_m,
        lcc_target=lcc_target,
        n_spline_points=n_spline_points,
    )

    # Check if LCC target reached
    conn_mid = connectivity_report(healed_graph)
    if conn_mid["lcc_pct"] >= lcc_target or not type_b_pairs:
        metrics = {
            "type_a_pairs":   len(type_a_pairs),
            "type_b_pairs":   len(type_b_pairs),
            "type_a_healed":  metrics_a.get("healed_edges", 0),
            "type_b_healed":  0,
            "lcc_before":     metrics_a.get("lcc_before", 0),
            "lcc_after":      conn_mid["lcc_pct"],
            "lcc_improvement": round(conn_mid["lcc_pct"] - metrics_a.get("lcc_before", 0), 4),
            "occlusion_guided": True,
        }
        return healed_graph, metrics

    # Phase 2: Conservative heal of Type B (genuine gap) breaks
    conservative_snap = snap_m * 0.5
    healed_graph2, metrics_b = bearing_aware_spline_bridge(
        healed_graph, type_b_pairs,
        max_heal_dist_m=conservative_snap,
        lcc_target=lcc_target,
        n_spline_points=n_spline_points,
    )

    conn_final = connectivity_report(healed_graph2)

    metrics = {
        "type_a_pairs":    len(type_a_pairs),
        "type_b_pairs":    len(type_b_pairs),
        "type_a_healed":   metrics_a.get("healed_edges", 0),
        "type_b_healed":   metrics_b.get("healed_edges", 0),
        "lcc_before":      metrics_a.get("lcc_before", 0),
        "lcc_after":       conn_final["lcc_pct"],
        "lcc_improvement": round(conn_final["lcc_pct"] - metrics_a.get("lcc_before", 0), 4),
        "occlusion_guided": True,
    }

    return healed_graph2, metrics


# ══════════════════════════════════════════════════════════════
# PRINT REPORT
# ══════════════════════════════════════════════════════════════

def print_sar_report(occlusion_metrics: Dict,
                     heal_metrics: Optional[Dict] = None,
                     sar_available: bool = True) -> None:
    """Print Phase 21 SAR integration report."""
    SEP = "─" * 60
    print(f"\n{SEP}")
    print(f"  PHASE 21 — SAR-GUIDED OCCLUSION INTEGRATION")
    print(SEP)

    if not sar_available:
        print(f"  ○ SAR mask not available — all breaks treated as Type A")
        print(f"  ○ (Provide sar_mask.npy from Sentinel-1 for full benefit)")
        print(f"\n{SEP}")
        print(f"  SAR: ○ FALLBACK MODE (no SAR data)")
        print(SEP)
        return

    print(f"  SAR source        : Sentinel-1 (synthetic fallback)")
    print(f"  Optical road px   : {occlusion_metrics['optical_road_px']:,}")
    print(f"  SAR road px       : {occlusion_metrics['sar_road_px']:,}")
    print(f"  Occlusion pixels  : {occlusion_metrics['occlusion_pixels']:,} "
          f"({occlusion_metrics['occlusion_pct']:.3f}% of image)")
    print(f"  SAR confirms      : {occlusion_metrics['sar_confirm_pixels']:,} px")
    print(f"  SAR misses        : {occlusion_metrics['sar_miss_pixels']:,} px")

    if heal_metrics:
        print(f"\n  Break classification:")
        print(f"    Type A (occluded) : {heal_metrics['type_a_pairs']} pairs → "
              f"{heal_metrics['type_a_healed']} healed (full snap)")
        print(f"    Type B (genuine)  : {heal_metrics['type_b_pairs']} pairs → "
              f"{heal_metrics['type_b_healed']} healed (half snap)")
        print(f"  LCC: {heal_metrics['lcc_before']:.1%} → {heal_metrics['lcc_after']:.1%}")

    print(f"\n{SEP}")
    print(f"  SAR: [OK] ACTIVE — occlusion-aware healing enabled")
    print(SEP)


# ══════════════════════════════════════════════════════════════
# TOP-LEVEL FUNCTION (called by run.py)
# ══════════════════════════════════════════════════════════════

def run_sar_integration(
        optical_mask: np.ndarray,
        affine: AffineTransform,
        sar_mask_path: Optional[str] = None,
) -> Tuple[np.ndarray, Dict, bool]:
    """
    Load or synthesise SAR mask and compute occlusion map.

    Parameters
    ----------
    optical_mask   : np.ndarray — from Phase 03
    affine         : AffineTransform — from Phase 03
    sar_mask_path  : str | None — path to sar_mask.npy from Part A
                     If None or not found, uses synthetic SAR

    Returns
    -------
    (occlusion_map, metrics, sar_available)
    """
    sar_available = False
    sar_mask = None

    if sar_mask_path and os.path.exists(sar_mask_path):
        try:
            sar_mask = np.load(sar_mask_path).astype(np.uint8)
            if sar_mask.shape != optical_mask.shape:
                print(f"  ⚠ SAR mask shape {sar_mask.shape} != "
                      f"optical {optical_mask.shape} — using synthetic")
                sar_mask = None
            else:
                sar_available = True
                print(f"  [OK] SAR mask loaded: {sar_mask_path}")
        except Exception as e:
            print(f"  ⚠ SAR mask load failed: {e} — using synthetic")

    if sar_mask is None:
        print(f"  ○ Generating synthetic SAR mask (no real SAR available)")
        sar_mask = make_synthetic_sar_mask(optical_mask)
        sar_available = False  # synthetic = fallback mode

    occlusion_map, metrics = compute_occlusion_map(optical_mask, sar_mask)
    return occlusion_map, metrics, sar_available

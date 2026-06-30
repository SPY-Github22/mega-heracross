"""
part_b_skeleton/osmnx_fallback.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 17: OSMnx Fallback Demo Mode

When shared/config.py has GRAPH_SOURCE = 'osmnx', this module
bypasses Part A's mask entirely and extracts the graph directly
from OpenStreetMap for the test tile.

Use cases:
  1. Hackathon demo when Part A's model isn't trained yet
  2. Baseline comparison — OSM is the theoretical ceiling F1=1.0
  3. Offline validation — test that Part C works before Part A delivers
  4. Quick prototype demo for judges before full pipeline is ready

The output is identical RoadGraph schema regardless of source.
All downstream phases (healing, simplification, audit, evaluation)
run exactly the same on osmnx-sourced graphs.

Source switching:
  GRAPH_SOURCE = 'part_b'  → use Part A mask (normal mode)
  GRAPH_SOURCE = 'osmnx'   → use OSM directly (fallback mode)

This is set in shared/config.py — never hardcoded here.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import sys
from typing import Tuple, Optional

# ── repo root on path ─────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from shared.schema import RoadGraph, RoadMaskMeta
from shared.config import (
    TARGET_CRS, GRAPH_SOURCE, TEST_TILE_BBOX,
    ROAD_MASK_PATH, META_PATH,
)
from part_b_skeleton.osm_reference import (
    load_or_download_osm, OSM_CACHE_PATH
)


# ══════════════════════════════════════════════════════════════
# SOURCE DETECTION
# ══════════════════════════════════════════════════════════════

def get_graph_source() -> str:
    """Return the active graph source from config (reads dynamically)."""
    from shared import config as _cfg
    return _cfg.GRAPH_SOURCE


def is_osmnx_mode() -> bool:
    """Return True if GRAPH_SOURCE = 'osmnx' (reads dynamically)."""
    return get_graph_source() == "osmnx"


# ══════════════════════════════════════════════════════════════
# OSMNX EXTRACTION
# ══════════════════════════════════════════════════════════════

def extract_from_osmnx(cache_path: str = OSM_CACHE_PATH,
                       force_refresh: bool = False
                       ) -> Tuple[RoadGraph, dict]:
    """
    Extract road graph directly from OSM for the test tile.
    Uses cached osm_reference.json if available.

    This is identical to what Phase 06 downloads for ground truth —
    in osmnx mode, the ground truth IS the graph, so F1 = 1.0.

    Returns
    -------
    (road_graph, stats) : (RoadGraph, dict)
    """
    road_graph, stats = load_or_download_osm(
        cache_path=cache_path,
        force_refresh=force_refresh,
        network_type="drive",
    )
    return road_graph, stats


def make_osmnx_meta() -> RoadMaskMeta:
    """
    Create a synthetic RoadMaskMeta for osmnx mode.
    Since there's no real mask, we synthesise plausible values
    consistent with the Koramangala test tile.
    """
    min_lon, min_lat, max_lon, max_lat = TEST_TILE_BBOX
    return RoadMaskMeta(
        crs         = TARGET_CRS,
        bbox        = TEST_TILE_BBOX,
        resolution_m= 5.8,      # LISS-IV native resolution
        source      = "osmnx",
    )


# ══════════════════════════════════════════════════════════════
# PRINT REPORT
# ══════════════════════════════════════════════════════════════

def print_osmnx_mode_banner() -> None:
    """Print a clear banner when running in osmnx fallback mode."""
    SEP = "═" * 60
    print(f"\n{SEP}")
    print(f"  ⚡ OSMNX FALLBACK MODE  (GRAPH_SOURCE='osmnx')")
    print(f"  Bypassing Part A mask — using OSM directly")
    print(f"  This gives theoretical ceiling performance (F1 ≈ 1.0)")
    print(f"  Switch to GRAPH_SOURCE='part_b' for real vision pipeline")
    print(SEP)


# ══════════════════════════════════════════════════════════════
# SCHEMA VALIDATION
# ══════════════════════════════════════════════════════════════

def validate_osmnx_graph(graph: RoadGraph) -> list:
    """
    Validate that the osmnx-extracted graph meets our contract.
    Same checks as validate_graph_basic in graph_builder.py.
    Returns list of violation strings.
    """
    violations = []

    if len(graph.nodes) == 0:
        violations.append("osmnx graph has zero nodes")
    if len(graph.edges) == 0:
        violations.append("osmnx graph has zero edges")
    if graph.crs != TARGET_CRS:
        violations.append(f"crs='{graph.crs}' must be '{TARGET_CRS}'")

    valid_ids = {n.id for n in graph.nodes}
    for i, e in enumerate(graph.edges):
        if e.source not in valid_ids:
            violations.append(f"edges[{i}].source={e.source} not in nodes")
        if e.target not in valid_ids:
            violations.append(f"edges[{i}].target={e.target} not in nodes")
        if e.weight_m <= 0:
            violations.append(f"edges[{i}].weight_m={e.weight_m} must be > 0")
        if len(e.geometry) < 2:
            violations.append(f"edges[{i}].geometry has < 2 points")

    return violations


# ══════════════════════════════════════════════════════════════
# TOP-LEVEL FUNCTION (called by run.py)
# ══════════════════════════════════════════════════════════════

def run_osmnx_fallback() -> Tuple[RoadGraph, RoadMaskMeta, dict]:
    """
    Full Phase 17 pipeline for osmnx mode:
      1. Print banner
      2. Extract graph from OSM (cached)
      3. Validate schema
      4. Return (graph, synthetic_meta, stats)

    The returned graph feeds directly into Phase 10 (healing)
    skipping Phases 03–05 (mask loading, skeletonization, sknw).

    Returns
    -------
    (road_graph, meta, stats)
    """
    print_osmnx_mode_banner()

    road_graph, stats = extract_from_osmnx()
    violations = validate_osmnx_graph(road_graph)

    if violations:
        print(f"  ⚠ osmnx graph violations ({len(violations)}):")
        for v in violations:
            print(f"    ✗ {v}")
    else:
        print(f"  [OK] osmnx graph valid: "
              f"{len(road_graph.nodes)} nodes, {len(road_graph.edges)} edges")

    meta = make_osmnx_meta()
    return road_graph, meta, stats

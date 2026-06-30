# part_a_vision/__init__.py
# Part A Vision Engine - entry point.
# This file runs when `from part_a_vision import ...` is called.
# It confirms shared contracts are accessible and prints engine info.

import os
import sys

# ── Repo root resolution ───────────────────────────────────────────────────────
_here      = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_here)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# ── Contract import check ──────────────────────────────────────────────────────
# If this fails, the shared/ folder is not set up correctly.
# The error message tells you exactly what to do.
try:
    from shared.schema import RoadMaskMeta, RoadGraph, GraphNode, GraphEdge
    from shared.config import (
        TARGET_CRS, TEST_TILE_BBOX,
        ROAD_MASK_PATH, META_PATH,
        GRAPH_PATH, HEATMAP_PATH
    )
    _contract_loaded = True
except ImportError as e:
    raise ImportError(
        f"\n[Part A] Cannot import shared contracts: {e}\n"
        f"  Expected location: {_repo_root}/shared/schema.py\n"
        f"  Expected location: {_repo_root}/shared/config.py\n"
        f"  Fix: Run scripts from the repo root:\n"
        f"       cd {_repo_root}\n"
        f"       python part_a_vision/..."
    ) from e

# ── Engine banner ──────────────────────────────────────────────────────────────
print("+------------------------------------------------------+")
print("|  Mega-Heracross - Part A: Vision & Occlusion Engine  |")
print("|  BAH 2026 · Problem Statement 4 · Route Resilience   |")
print("+------------------------------------------------------+")
print(f"  Contract : road_mask.npy (uint8, HxW, {{0,1}}) + meta.json")
print(f"  CRS      : {TARGET_CRS}")
print(f"  Test tile: Koramangala {TEST_TILE_BBOX}")
print(f"  Output   : {ROAD_MASK_PATH}")
print(f"  Output   : {META_PATH}")
print()
"""
part_b_skeleton/resolution_config.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 18: Multi-Resolution Mask Support

All algorithm thresholds in Part B are physically motivated and
must scale with the sensor's ground sampling distance (GSD).

Supported sensors (from meta.json source field):
  LISS-IV    : 5.8 m/px  (primary ISRO sensor for this problem)
  Sentinel-2 : 10.0 m/px (backup optical)
  Sentinel-1 : 10.0 m/px (SAR — same GSD as S2)
  synthetic  : variable   (test data)

Every threshold derives from resolution_m via physical reasoning:
  snap_m      = gap_px × resolution_m × 1.5
              (heal breaks up to 1.5× the typical occlusion gap size)

  stub_m      = resolution_m × 1.5
              (prune stubs shorter than 1.5 pixels)

  min_road_length_m = resolution_m × 3
              (smallest blob considered a real road)

  close_radius_px = max(1, round(resolution_m / 5.8))
              (morphological closing scales with resolution)

  width_stub_threshold_m = resolution_m × 8
              (width violation tolerance at junctions)

Known sensor defaults (used when source matches):
  LISS-IV    → snap=34.8m, stub=8.7m,  gap=23.2m
  Sentinel-2 → snap=30.0m, stub=15.0m, gap=20.0m
  synthetic  → snap computed from actual resolution_m
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

# ── repo root on path ─────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ══════════════════════════════════════════════════════════════
# KNOWN SENSOR PROFILES
# ══════════════════════════════════════════════════════════════

# Maps source string (from meta.json) → canonical resolution_m
# Used to override resolution_m if meta.json has an incorrect value
_SENSOR_RESOLUTION = {
    "liss-iv":    5.8,
    "lissiv":     5.8,
    "liss_iv":    5.8,
    "sentinel-2": 10.0,
    "sentinel2":  10.0,
    "s2":         10.0,
    "sentinel-1": 10.0,
    "sentinel1":  10.0,
    "s1":         10.0,
    "sar":        10.0,
}

# Typical occlusion gap size in pixels per sensor
# (tree canopy / cloud shadow typical width)
_SENSOR_GAP_PX = {
    "liss-iv":    4,   # ~23m at 5.8m/px
    "sentinel-2": 2,   # ~20m at 10m/px
    "sentinel-1": 2,   # SAR has fewer occlusion gaps
}


def _normalise_source(source: str) -> str:
    """Normalise source string to lowercase, strip spaces."""
    return source.lower().strip().replace(" ", "-")


# ══════════════════════════════════════════════════════════════
# RESOLUTION-AWARE CONFIG
# ══════════════════════════════════════════════════════════════

@dataclass
class ResolutionAwareConfig:
    """
    All algorithm parameters derived from sensor resolution.

    Instantiate with resolution_m (from meta.json) and optionally
    the source string (to apply sensor-specific tuning).

    All parameters are computed once at construction and exposed
    as attributes. No hidden magic — every value is logged.

    Attributes
    ----------
    resolution_m          : float — ground sampling distance in m/px
    source                : str   — sensor name (for logging)

    snap_m                : float — break detection radius (healing)
    second_pass_snap_m    : float — relaxed radius for 2nd healing pass
    stub_m                : float — minimum stub length to keep
    min_road_length_m     : float — minimum blob area for preprocessing
    close_radius_px       : int   — morphological closing footprint
    width_stub_threshold_m: float — junction width tolerance

    typical_gap_m         : float — expected occlusion gap size
    road_width_m          : float — expected road width in real world
    """
    resolution_m: float
    source: str = "unknown"

    # Computed at post-init
    snap_m:                 float = field(init=False)
    second_pass_snap_m:     float = field(init=False)
    stub_m:                 float = field(init=False)
    min_road_length_m:      float = field(init=False)
    close_radius_px:        int   = field(init=False)
    width_stub_threshold_m: float = field(init=False)
    typical_gap_m:          float = field(init=False)
    road_width_m:           float = field(init=False)
    effective_resolution_m: float = field(init=False)

    def __post_init__(self):
        src_norm = _normalise_source(self.source)

        # Use sensor-specific resolution if known (overrides meta.json)
        # This handles cases where meta.json has a slightly wrong value
        if src_norm in _SENSOR_RESOLUTION:
            self.effective_resolution_m = _SENSOR_RESOLUTION[src_norm]
        else:
            self.effective_resolution_m = self.resolution_m

        res = self.effective_resolution_m

        # Typical occlusion gap in pixels (sensor-specific or derived)
        if src_norm in _SENSOR_GAP_PX:
            gap_px = _SENSOR_GAP_PX[src_norm]
        else:
            # For unknown sensors: gap is always 4 pixels
            # (conservative — heals up to 4-pixel occlusion gaps)
            # snap_m then scales linearly with resolution_m
            gap_px = 4

        self.typical_gap_m          = gap_px * res
        self.snap_m                 = round(self.typical_gap_m * 1.5, 1)
        self.second_pass_snap_m     = round(self.snap_m * 1.75, 1)
        self.stub_m                 = round(max(5.0, res * 1.5), 1)
        self.min_road_length_m      = round(max(10.0, res * 3.0), 1)
        self.close_radius_px        = max(1, round(res / 5.8))
        self.width_stub_threshold_m = round(res * 8.0, 1)

        # Typical road width for Bengaluru urban roads
        # (2-lane road ≈ 7m real world)
        self.road_width_m = round(max(7.0, res * 2.0), 1)

    def log(self) -> str:
        """Return a formatted parameter table for logging."""
        lines = [
            f"  Resolution-Aware Config ({self.source})",
            f"    Sensor resolution    : {self.resolution_m:.1f} m/px",
            f"    Effective resolution : {self.effective_resolution_m:.1f} m/px",
            f"    Typical occlusion gap: {self.typical_gap_m:.1f} m",
            f"    Snap radius (pass 1) : {self.snap_m:.1f} m",
            f"    Snap radius (pass 2) : {self.second_pass_snap_m:.1f} m",
            f"    Stub prune threshold : {self.stub_m:.1f} m",
            f"    Min road length      : {self.min_road_length_m:.1f} m",
            f"    Morph close radius   : {self.close_radius_px} px",
            f"    Road width estimate  : {self.road_width_m:.1f} m",
        ]
        return "\n".join(lines)

    def summary_line(self) -> str:
        """One-line summary for judge report."""
        return (f"res={self.effective_resolution_m:.1f}m/px  "
                f"snap={self.snap_m:.0f}m  "
                f"stub={self.stub_m:.0f}m  "
                f"gap={self.typical_gap_m:.0f}m")


# ══════════════════════════════════════════════════════════════
# FACTORY FUNCTION
# ══════════════════════════════════════════════════════════════

def make_config(resolution_m: float,
                source: str = "unknown") -> ResolutionAwareConfig:
    """
    Create a ResolutionAwareConfig from resolution_m and source.
    This is the single entry point used by all pipeline stages.

    Parameters
    ----------
    resolution_m : float — from meta.json
    source       : str   — from meta.json (e.g. 'LISS-IV', 'Sentinel-2')

    Returns
    -------
    ResolutionAwareConfig
    """
    return ResolutionAwareConfig(resolution_m=resolution_m, source=source)


# ══════════════════════════════════════════════════════════════
# PRINT REPORT
# ══════════════════════════════════════════════════════════════

def print_resolution_config(cfg: ResolutionAwareConfig) -> None:
    """Print Phase 18 resolution config report."""
    SEP = "─" * 60
    print(f"\n{SEP}")
    print(f"  PHASE 18 — RESOLUTION-AWARE CONFIG")
    print(SEP)
    print(cfg.log())

    # Sensor identification
    src = _normalise_source(cfg.source)
    if src in _SENSOR_RESOLUTION:
        print(f"\n  ✓ Known sensor: {cfg.source} "
              f"(official GSD = {_SENSOR_RESOLUTION[src]:.1f} m/px)")
    else:
        print(f"\n  ○ Unknown sensor '{cfg.source}' — "
              f"using meta.json resolution_m={cfg.resolution_m:.1f}")

    print(f"\n  All pipeline thresholds derived from resolution:")
    print(f"    {cfg.summary_line()}")
    print(f"\n{SEP}")
    print(f"  RESOLUTION CONFIG: ✓ READY")
    print(SEP)

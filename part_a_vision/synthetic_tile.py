# part_a_vision/synthetic_tile.py
#
# Synthetic Tile Generator for Mega-Heracross Part A.
#
# What this file does:
#   1. Downloads the real Koramangala OSMnx road graph (once, then caches it)
#   2. Rasterizes that graph onto a (H, W) binary ground truth mask
#   3. Places synthetic buildings and vegetation around the road network
#   4. Generates a 4-band LISS-IV-like optical tile (Green, Red, NIR, SWIR)
#   5. Generates a 2-band Sentinel-1-like SAR tile (VV, VH)
#   6. Optionally applies occlusion (cloud, canopy, building shadow)
#   7. Saves demo output via output_writer.py (road_mask.npy + meta.json)
#   8. Saves test tiles for Phases 3-6
#   9. Generates visualization PNGs
#
# Why we need this before having real satellite data:
#   Parts B and C need our road_mask.npy to test their code NOW.
#   Training (Phase 7) needs labeled tiles NOW.
#   This generator fills both needs until real LISS-IV data arrives.
#
# Usage:
#   python part_a_vision/synthetic_tile.py --mode demo
#   python part_a_vision/synthetic_tile.py --mode dataset --n_train 5 --n_val 2
#   python part_a_vision/synthetic_tile.py --mode demo --occlusion cloud --fraction 0.4

import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

# Non-interactive backend - prevents matplotlib from trying to open a window
# on Windows when running from the terminal. Must be set BEFORE pyplot import.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Repo root on sys.path ──────────────────────────────────────────────────────
_here      = Path(__file__).resolve().parent
_repo_root = _here.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from shared.config import TARGET_CRS, TEST_TILE_BBOX
from part_a_vision.part_a_config import (
    TILE_SIZE, SYNTHETIC_RESOLUTION_M, SYNTHETIC_SEED,
    SYNTHETIC_OCCLUSION_MAX, OSMNX_CACHE_PATH, OSMNX_GT_MASK_PATH,
)
from part_a_vision.output_writer import write_road_mask, write_meta, load_and_verify


# ------------------------------------------------------------------------------
# SPECTRAL SIGNATURES
# ------------------------------------------------------------------------------
#
# These values approximate the spectral reflectance of each land cover type
# as seen by LISS-IV's 4 bands, normalized to [0, 1].
#
# LISS-IV band order we use: [Green (B2), Red (B3), NIR (B4), SWIR (B5)]
# Source: LISS-IV sensor characteristics + urban RS literature
#
# Why do roads have a flat, dark spectrum?
#   Asphalt absorbs most incoming radiation across all wavelengths.
#   No strong absorption features → flat spectrum at low reflectance (~0.10-0.15).
#
# Why does vegetation have very high NIR (index 2)?
#   Plant cells scatter NIR strongly (internal leaf structure scattering).
#   This is the physical basis of NDVI. High NIR + low Red = green vegetation.

SPECTRAL = {
    # [Green,  Red,   NIR,   SWIR]
    'background': np.array([0.200, 0.250, 0.300, 0.350], dtype=np.float32),
    'road':       np.array([0.120, 0.120, 0.120, 0.140], dtype=np.float32),
    'vegetation': np.array([0.140, 0.075, 0.550, 0.190], dtype=np.float32),
    'building':   np.array([0.440, 0.440, 0.390, 0.470], dtype=np.float32),
    'cloud':      np.array([0.860, 0.870, 0.840, 0.810], dtype=np.float32),
}
# Per-band texture noise std (how much natural variation each class has)
SPECTRAL_NOISE_STD = {
    'background': np.array([0.04, 0.04, 0.05, 0.05], dtype=np.float32),
    'road':       np.array([0.02, 0.02, 0.02, 0.02], dtype=np.float32),
    'vegetation': np.array([0.03, 0.02, 0.08, 0.04], dtype=np.float32),
    'building':   np.array([0.06, 0.06, 0.05, 0.06], dtype=np.float32),
    'cloud':      np.array([0.03, 0.03, 0.02, 0.03], dtype=np.float32),
}

# SAR backscatter values, normalized [0, 1]
# [VV, VH] polarization
#
# Why are roads dark in SAR?
#   Smooth asphalt behaves like a specular reflector (like a mirror).
#   Most of the radar signal bounces away from the antenna → low return.
#   This is the OPPOSITE of what you might expect from the name "road detection."
#
# Why are buildings bright?
#   Building walls + ground create a "double-bounce" corner reflector.
#   Radar hits the ground, bounces off the wall, returns directly to antenna.
#   Very high return signal → bright in SAR imagery.

SAR_BACKSCATTER = {
    # [VV,   VH]
    'background': np.array([0.350, 0.200], dtype=np.float32),
    'road':       np.array([0.075, 0.050], dtype=np.float32),
    'vegetation': np.array([0.500, 0.440], dtype=np.float32),
    'building':   np.array([0.820, 0.640], dtype=np.float32),
}
SAR_NOISE_STD = {
    'background': np.array([0.08, 0.06], dtype=np.float32),
    'road':       np.array([0.02, 0.02], dtype=np.float32),
    'vegetation': np.array([0.12, 0.10], dtype=np.float32),
    'building':   np.array([0.10, 0.08], dtype=np.float32),
}

# Pixel width used when drawing roads onto the mask.
# At 5.8m/pixel, a 2-pixel road = ~11.6m wide (typical 2-lane Indian road).
# This is thin enough to be challenging for the model - exactly what we want.
ROAD_WIDTH_PX = 2


# ------------------------------------------------------------------------------
# DATA CONTAINERS
# ------------------------------------------------------------------------------

@dataclass
class SyntheticTile:
    """
    One complete synthetic tile with all its components.
    The model sees (optical + sar) as input and tries to predict gt_mask.
    """
    optical:          np.ndarray   # (4, H, W) float32 - model input (optical bands)
    sar:              np.ndarray   # (2, H, W) float32 - model input (SAR bands)
    gt_mask:          np.ndarray   # (H, W) uint8      - ground truth (road=1, bg=0)
    cloud_mask:       np.ndarray   # (H, W) uint8      - 1=cloud/occlusion, 0=clear
    occlusion_type:   str          # "none" | "cloud" | "canopy" | "shadow" | "all"
    occlusion_fraction: float      # fraction of tile that is occluded [0, 1]
    seed:             int
    bbox:             tuple        # (min_lon, min_lat, max_lon, max_lat)


# ------------------------------------------------------------------------------
# MAIN GENERATOR CLASS
# ------------------------------------------------------------------------------

class SyntheticTileGenerator:
    """
    Generates synthetic satellite tiles for Koramangala, Bengaluru.

    Core idea:
        1. Use REAL OSMnx road topology (cached from OpenStreetMap)
        2. Synthesize plausible optical + SAR textures around that topology
        3. Apply realistic occlusion patterns
        4. Ground truth is always the clean road mask (occlusion-free)

    Why real road topology?
        A judge who knows Bengaluru will recognize Koramangala's actual
        road structure. Procedural roads look fake; real OSMnx does not.
    """

    def __init__(
        self,
        bbox:             tuple = TEST_TILE_BBOX,
        H:                int   = TILE_SIZE,
        W:                int   = TILE_SIZE,
        resolution_m:     float = SYNTHETIC_RESOLUTION_M,
        osmnx_cache_path: str   = OSMNX_CACHE_PATH,
        road_width_px:    int   = ROAD_WIDTH_PX,
    ):
        self.bbox             = bbox
        self.H                = H
        self.W                = W
        self.resolution_m     = resolution_m
        self.osmnx_cache_path = Path(osmnx_cache_path)
        self.road_width_px    = road_width_px
        self.G                = None   # OSMnx graph - loaded on first use
        self._gt_mask_cache   = None  # cached road mask (same for all tiles)

        print(f"[SyntheticTileGenerator] Initialized")
        print(f"  Tile size:   {H}x{W} pixels")
        print(f"  Resolution:  {resolution_m}m/pixel")
        print(f"  BBox:        {bbox}")

    # ── OSMnx Graph ───────────────────────────────────────────────────────────

    def _load_or_download_osmnx(self) -> None:
        """
        Load the Koramangala road graph from cache file if it exists,
        otherwise download it from OpenStreetMap and save to cache.

        This function is idempotent: calling it multiple times is safe.
        The cache means no internet is needed after the first run -
        critical for the offline demo at NRSC Hyderabad.
        """
        try:
            import osmnx as ox
        except ImportError:
            raise ImportError(
                "[SyntheticTileGenerator] osmnx is not installed.\n"
                "  Fix: pip install osmnx"
            )

        # Suppress osmnx download progress bars (noisy in a pipeline)
        ox.settings.log_console = False

        cache_path = self.osmnx_cache_path
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if cache_path.exists():
            print(f"  Loading OSMnx graph from cache: {cache_path}")
            self.G = ox.load_graphml(str(cache_path))
            n_nodes = len(self.G.nodes)
            n_edges = len(self.G.edges)
            print(f"  Graph loaded: {n_nodes} nodes, {n_edges} edges")
        else:
            print(f"  Downloading Koramangala road graph from OpenStreetMap...")
            print(f"  (This happens only once. Saved to {cache_path})")
            min_lon, min_lat, max_lon, max_lat = self.bbox

            # Handle OSMnx API differences between v1.x and v2.x
            # osmnx 2.0 changed the bbox parameter format.
            try:
                # osmnx >= 2.0: bbox as (left, bottom, right, top) tuple
                self.G = ox.graph_from_bbox(
                    bbox=(min_lon, min_lat, max_lon, max_lat),
                    network_type='drive',
                    retain_all=False,
                    truncate_by_edge=True,
                )
            except TypeError:
                try:
                    # osmnx 1.x: positional (north, south, east, west)
                    self.G = ox.graph_from_bbox(
                        max_lat, min_lat, max_lon, min_lon,
                        network_type='drive',
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"[SyntheticTileGenerator] OSMnx download failed: {e}\n"
                        f"  Check your internet connection and osmnx version."
                    )

            ox.save_graphml(self.G, str(cache_path))
            n_nodes = len(self.G.nodes)
            n_edges = len(self.G.edges)
            print(f"  Downloaded and cached: {n_nodes} nodes, {n_edges} edges")

    # ── Road Rasterization ────────────────────────────────────────────────────

    def _lonlat_to_pixel(self, lon: float, lat: float) -> Tuple[int, int]:
        """
        Convert geographic (lon, lat) in EPSG:4326 to pixel (col, row).

        Note the Y-axis flip: in images, row 0 is the TOP of the image.
        But latitude INCREASES going up (north). So:
            row = (max_lat - lat) / lat_span * H
        A point at max_lat (northernmost) → row 0 (top of image). Correct.
        A point at min_lat (southernmost) → row H-1 (bottom). Correct.
        """
        min_lon, min_lat, max_lon, max_lat = self.bbox
        col = int(round((lon - min_lon) / (max_lon - min_lon) * (self.W - 1)))
        row = int(round((max_lat - lat) / (max_lat - min_lat) * (self.H - 1)))
        col = max(0, min(self.W - 1, col))
        row = max(0, min(self.H - 1, row))
        return col, row  # (x, y) for cv2

    def _rasterize_roads(self) -> np.ndarray:
        """
        Draw each OSMnx edge as a polyline onto a binary (H, W) mask.

        Each edge has an optional 'geometry' attribute (a Shapely LineString
        with intermediate points for curved roads). If absent, we fall back
        to a straight line between the two endpoint nodes.

        Returns:
            gt_mask: (H, W) uint8, values in {0, 1}
        """
        if self.G is None:
            self._load_or_download_osmnx()

        mask = np.zeros((self.H, self.W), dtype=np.uint8)
        edges_drawn = 0

        for u, v, data in self.G.edges(data=True):
            # Get the coordinate sequence for this road segment
            if 'geometry' in data:
                # Shapely LineString → list of (lon, lat) tuples
                coords = list(data['geometry'].coords)
            else:
                # No geometry stored - straight line between nodes
                u_node = self.G.nodes[u]
                v_node = self.G.nodes[v]
                coords = [
                    (u_node['x'], u_node['y']),
                    (v_node['x'], v_node['y']),
                ]

            # Convert to pixel coordinates
            pixels = []
            for lon, lat in coords:
                col, row = self._lonlat_to_pixel(lon, lat)
                pixels.append([col, row])

            if len(pixels) < 2:
                continue

            # cv2.polylines expects shape (N, 1, 2) int32
            pts = np.array(pixels, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                mask, [pts],
                isClosed=False,
                color=1,
                thickness=self.road_width_px,
                lineType=cv2.LINE_AA,
            )
            edges_drawn += 1

        road_px    = int(mask.sum())
        total_px   = self.H * self.W
        road_frac  = road_px / total_px * 100
        print(f"  Rasterized {edges_drawn} road edges → "
              f"{road_px:,} road pixels ({road_frac:.1f}% of tile)")

        return mask

    def _get_gt_mask(self) -> np.ndarray:
        """
        Return the road ground truth mask, loading from disk cache if available.
        The GT mask is the same for all generated tiles (it's fixed by OSMnx).
        Only the textures and occlusions change between tiles.
        """
        if self._gt_mask_cache is not None:
            return self._gt_mask_cache.copy()

        gt_cache = Path(OSMNX_GT_MASK_PATH)

        if gt_cache.exists():
            self._gt_mask_cache = np.load(str(gt_cache))
            print(f"  GT mask loaded from cache: {gt_cache}")
        else:
            print(f"  Generating road GT mask from OSMnx graph...")
            self._gt_mask_cache = self._rasterize_roads()
            gt_cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(gt_cache), self._gt_mask_cache)
            print(f"  GT mask cached: {gt_cache}")

        return self._gt_mask_cache.copy()

    # ── Scene Layout ──────────────────────────────────────────────────────────

    def _multiscale_texture(self, rng: np.random.Generator, H: int, W: int) -> np.ndarray:
        """
        Generate a spatially-correlated noise texture using multi-scale Gaussian
        filtering. This approximates Perlin noise without any extra dependencies.

        Idea: start with white noise, smooth at multiple scales, weighted sum.
        - Large sigma (20) → gentle low-frequency terrain undulation
        - Medium sigma (5) → block-level texture variation (urban patches)
        - Small sigma (1) → fine-grain material texture

        Returns: (H, W) float32 in [0, 1]
        """
        noise = rng.standard_normal((H, W)).astype(np.float32)
        texture = (
            0.50 * gaussian_filter(noise, sigma=20) +
            0.35 * gaussian_filter(noise, sigma=5)  +
            0.15 * gaussian_filter(noise, sigma=1)
        )
        # Normalize to [0, 1]
        lo, hi = texture.min(), texture.max()
        texture = (texture - lo) / (hi - lo + 1e-8)
        return texture

    def _generate_buildings(
        self,
        rng:       np.random.Generator,
        road_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Place synthetic building footprints (rectangular) in non-road areas.

        Strategy:
        - Dilate road mask by 5px to create a no-build buffer zone around roads
        - Try to place 100-160 random rectangles in the remaining space
        - Skip any rectangle that overlaps an existing building or the road buffer
        - Building sizes reflect Koramangala's dense urban fabric:
          width 8-35px (~46-200m), height 10-45px (~58-260m)

        Returns: (H, W) uint8 buildings mask
        """
        H, W = self.H, self.W
        buildings = np.zeros((H, W), dtype=np.uint8)

        # Exclusion zone: roads + 5px buffer around roads
        road_buffer = cv2.dilate(
            road_mask, np.ones((5, 5), dtype=np.uint8), iterations=2
        )

        # available[y,x] = 1 means we can place a building at (y,x)
        available = (1 - road_buffer).astype(np.uint8)

        n_target = int(rng.integers(100, 160))
        n_placed  = 0
        n_attempts = n_target * 4  # allow several attempts per target building

        for _ in range(n_attempts):
            if n_placed >= n_target:
                break

            # Random building dimensions (in pixels)
            bw = int(rng.integers(8, 36))    # building width
            bh = int(rng.integers(10, 46))   # building height

            if bw >= W - 2 or bh >= H - 2:
                continue

            # Random top-left corner
            x = int(rng.integers(1, W - bw - 1))
            y = int(rng.integers(1, H - bh - 1))

            # Check: entire candidate region must be available
            region_available = available[y:y+bh, x:x+bw]
            if region_available.min() == 0:
                continue

            # Place building
            buildings[y:y+bh, x:x+bw] = 1
            n_placed += 1

            # Update available: exclude building + 3px gap around it (alley)
            gap = 3
            y0, y1 = max(0, y - gap), min(H, y + bh + gap)
            x0, x1 = max(0, x - gap), min(W, x + bw + gap)
            available[y0:y1, x0:x1] = 0

        return buildings

    def _generate_vegetation(
        self,
        rng:           np.random.Generator,
        road_mask:     np.ndarray,
        buildings:     np.ndarray,
    ) -> np.ndarray:
        """
        Place synthetic vegetation patches (circular) in areas without roads or buildings.

        Two types:
        1. Large park patches (radius 20-60px): 2-5 per tile
           These represent Koramangala's scattered parks and tree-cover zones.
        2. Small tree clusters (radius 3-12px): 20-50 per tile
           These represent individual trees and small groups.

        Returns: (H, W) uint8 vegetation mask
        """
        H, W = self.H, self.W
        veg   = np.zeros((H, W), dtype=np.uint8)

        # Areas where vegetation cannot go (roads + buildings)
        occupied = np.maximum(road_mask, buildings)

        y_grid, x_grid = np.ogrid[:H, :W]

        # ── Large park patches ────────────────────────────────────────────────
        n_parks = int(rng.integers(2, 6))
        for _ in range(n_parks):
            cx     = int(rng.integers(30, W - 30))
            cy     = int(rng.integers(30, H - 30))
            radius = int(rng.integers(20, 61))

            dist = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
            park = (dist < radius).astype(np.uint8)
            park[occupied > 0] = 0  # cut out roads and buildings
            veg = np.maximum(veg, park)

        # ── Small scattered tree clusters ─────────────────────────────────────
        n_clusters = int(rng.integers(20, 51))
        for _ in range(n_clusters):
            cx     = int(rng.integers(0, W))
            cy     = int(rng.integers(0, H))
            radius = int(rng.integers(3, 13))

            dist    = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
            cluster = (dist < radius).astype(np.uint8)
            cluster[occupied > 0] = 0
            veg = np.maximum(veg, cluster)

        return veg

    # ── Texture Generation ────────────────────────────────────────────────────

    def _generate_optical(
        self,
        gt_mask:   np.ndarray,
        buildings: np.ndarray,
        veg:       np.ndarray,
        rng:       np.random.Generator,
    ) -> np.ndarray:
        """
        Generate a 4-band LISS-IV-like optical tile.

        Band order: [Green (0), Red (1), NIR (2), SWIR (3)]

        Strategy:
        1. Start with background spectral signature + low-frequency texture variation
        2. Paint vegetation pixels with vegetation spectral signature
        3. Paint building pixels (on top of vegetation if overlap)
        4. Paint road pixels (on top of everything - roads are most definite)
        5. Add fine-grain per-pixel noise

        The layering order matters: roads win over buildings, buildings win over veg.

        Returns: (4, H, W) float32, values in [0, 1]
        """
        H, W    = self.H, self.W
        optical = np.zeros((4, H, W), dtype=np.float32)

        # Base spatial texture (low-frequency terrain variation)
        base_texture = self._multiscale_texture(rng, H, W)  # (H, W) in [0,1]

        # Layer 1: Background
        for b in range(4):
            sig   = SPECTRAL['background'][b]
            std   = SPECTRAL_NOISE_STD['background'][b]
            optical[b] = sig + std * (base_texture - 0.5) * 2.0

        # Layer 2: Vegetation (overwrite background where veg exists)
        if veg.any():
            veg_noise = rng.normal(0, 1, (4, H, W)).astype(np.float32)
            veg_mask_b = veg.astype(bool)
            for b in range(4):
                sig = SPECTRAL['vegetation'][b]
                std = SPECTRAL_NOISE_STD['vegetation'][b]
                optical[b][veg_mask_b] = sig + std * veg_noise[b][veg_mask_b]

        # Layer 3: Buildings (overwrite background/veg)
        if buildings.any():
            bld_noise = rng.normal(0, 1, (4, H, W)).astype(np.float32)
            bld_mask_b = buildings.astype(bool)
            for b in range(4):
                sig = SPECTRAL['building'][b]
                std = SPECTRAL_NOISE_STD['building'][b]
                optical[b][bld_mask_b] = sig + std * bld_noise[b][bld_mask_b]

        # Layer 4: Roads (overwrite everything - roads are the definite class)
        if gt_mask.any():
            road_noise = rng.normal(0, 1, (4, H, W)).astype(np.float32)
            road_mask_b = gt_mask.astype(bool)
            for b in range(4):
                sig = SPECTRAL['road'][b]
                std = SPECTRAL_NOISE_STD['road'][b]
                optical[b][road_mask_b] = sig + std * road_noise[b][road_mask_b]

        # Fine-grain sensor noise across all pixels
        sensor_noise = rng.normal(0, 0.008, (4, H, W)).astype(np.float32)
        optical += sensor_noise

        return np.clip(optical, 0.0, 1.0)

    def _generate_sar(
        self,
        gt_mask:   np.ndarray,
        buildings: np.ndarray,
        veg:       np.ndarray,
        rng:       np.random.Generator,
    ) -> np.ndarray:
        """
        Generate a 2-band Sentinel-1-like SAR tile.

        Band order: [VV (0), VH (1)]

        Key physics encoded here:
        - Roads: dark (specular reflection away from antenna)
        - Buildings: bright (double-bounce corner reflector)
        - Vegetation: medium, textured (volume scattering)
        - Background: medium, slightly lower than vegetation

        Speckle noise:
        - Real SAR has multiplicative speckle from coherent signal interference.
        - We simulate this with Rayleigh-distributed noise (single-look approximation).
        - Formula: observed = true_backscatter * rayleigh_noise
        - Light Gaussian smoothing after speckle = simulates multi-look processing.

        Returns: (2, H, W) float32, values in [0, 1]
        """
        H, W = self.H, self.W
        sar  = np.zeros((2, H, W), dtype=np.float32)

        # Base SAR texture (different frequency from optical)
        for b in range(2):
            base = self._multiscale_texture(rng, H, W)
            sar[b] = SAR_BACKSCATTER['background'][b] * (0.6 + 0.8 * base)

        # Vegetation
        if veg.any():
            veg_noise = rng.normal(0, 1, (2, H, W)).astype(np.float32)
            veg_mask_b = veg.astype(bool)
            for b in range(2):
                sig = SAR_BACKSCATTER['vegetation'][b]
                std = SAR_NOISE_STD['vegetation'][b]
                sar[b][veg_mask_b] = sig + std * veg_noise[b][veg_mask_b]

        # Buildings (double-bounce - very bright)
        if buildings.any():
            bld_noise = rng.normal(0, 1, (2, H, W)).astype(np.float32)
            bld_mask_b = buildings.astype(bool)
            for b in range(2):
                sig = SAR_BACKSCATTER['building'][b]
                std = SAR_NOISE_STD['building'][b]
                sar[b][bld_mask_b] = sig + std * bld_noise[b][bld_mask_b]

        # Roads (specular - very dark)
        if gt_mask.any():
            road_noise = rng.normal(0, 1, (2, H, W)).astype(np.float32)
            road_mask_b = gt_mask.astype(bool)
            for b in range(2):
                sig = SAR_BACKSCATTER['road'][b]
                std = SAR_NOISE_STD['road'][b]
                sar[b][road_mask_b] = sig + std * road_noise[b][road_mask_b]

        # Multiplicative Rayleigh speckle noise
        # Rayleigh distribution approximates single-look SAR speckle
        # scale=0.10 → mean speckle amplitude of ~0.125 (moderate speckle)
        for b in range(2):
            speckle = rng.rayleigh(scale=0.10, size=(H, W)).astype(np.float32)
            sar[b] = sar[b] * speckle

        # Light Gaussian blur = simulates multi-look processing (reduces speckle)
        for b in range(2):
            sar[b] = gaussian_filter(sar[b], sigma=1.2).astype(np.float32)

        return np.clip(sar, 0.0, 1.0)

    # ── Occlusion Modes ───────────────────────────────────────────────────────

    def _apply_cloud_occlusion(
        self,
        optical:  np.ndarray,
        sar:      np.ndarray,
        fraction: float,
        rng:      np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply cloud cover to optical bands only.

        Physics:
        - Clouds are optically opaque → completely replace optical signal
        - Clouds have high reflectance across all bands (bright white appearance)
        - Clouds are NOT visible in SAR - radar sees THROUGH cloud cover
          (this is the primary motivation for SAR fusion in Part A)

        Implementation:
        - Generate large Gaussian blobs until we cover `fraction` of the image
        - Cloud spectral signature: near-uniform bright values [0.81-0.87]
        - SAR is returned unchanged

        Returns: occluded_optical, unchanged_sar, cloud_mask
        """
        H, W    = self.H, self.W
        total_px = H * W
        target_px = int(total_px * min(fraction, SYNTHETIC_OCCLUSION_MAX))

        cloud_mask = np.zeros((H, W), dtype=np.uint8)
        y_grid, x_grid = np.ogrid[:H, :W]

        attempts = 0
        while cloud_mask.sum() < target_px and attempts < 50:
            attempts += 1
            cx     = int(rng.integers(0, W))
            cy     = int(rng.integers(0, H))
            # Large blobs: radius 40-120px
            radius = float(rng.integers(40, 121))

            dist_sq = (x_grid - cx).astype(np.float32)**2 + \
                      (y_grid - cy).astype(np.float32)**2
            # Gaussian blob: pixels within ~1σ of radius are "cloud"
            blob = np.exp(-dist_sq / (2.0 * (radius * 0.5)**2))
            cloud_mask = np.maximum(cloud_mask, (blob > 0.20).astype(np.uint8))

        # Apply cloud spectral signature where cloud_mask == 1
        cloud_optical = optical.copy()
        cloud_noise   = rng.normal(0, 0.025, (4, H, W)).astype(np.float32)
        for b in range(4):
            cloud_optical[b] = np.where(
                cloud_mask > 0,
                SPECTRAL['cloud'][b] + cloud_noise[b],
                optical[b]
            )

        # SAR is physically unaffected by clouds - intentionally return unchanged
        cloud_sar = sar.copy()

        actual_frac = float(cloud_mask.sum()) / total_px
        return np.clip(cloud_optical, 0.0, 1.0), cloud_sar, cloud_mask, actual_frac

    def _apply_canopy_occlusion(
        self,
        optical:  np.ndarray,
        sar:      np.ndarray,
        gt_mask:  np.ndarray,
        fraction: float,
        rng:      np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply tree canopy patches over road pixels in optical AND partially in SAR.

        Physics:
        - Tree canopy in optical: replaces road pixel signature with vegetation signature
        - Tree canopy in SAR: partially attenuates signal (70% of original remains)
          SAR penetrates thin vegetation partially - this is different from clouds.
          Dense canopy can reduce SAR road contrast by 20-40%.

        Strategy:
        - Find road pixels → use them as cluster centers for canopy patches
        - Place circular canopy blobs on and around road pixels
        - Stop when we've covered `fraction` of total road pixels

        Returns: occluded_optical, partially_occluded_sar, canopy_mask
        """
        H, W      = self.H, self.W
        canopy    = np.zeros((H, W), dtype=np.uint8)
        road_pixels = np.argwhere(gt_mask > 0)  # (N, 2) array of [row, col]

        if len(road_pixels) == 0:
            return optical.copy(), sar.copy(), canopy, 0.0

        road_px_count = len(road_pixels)
        target_covered = int(road_px_count * min(fraction, SYNTHETIC_OCCLUSION_MAX))

        y_grid, x_grid = np.ogrid[:H, :W]
        attempts = 0

        while int((canopy * gt_mask).sum()) < target_covered and attempts < 100:
            attempts += 1
            # Cluster center: random road pixel
            idx    = int(rng.integers(0, road_px_count))
            cy, cx = road_pixels[idx]
            radius = float(rng.integers(12, 38))

            dist   = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2).astype(np.float32)
            blob   = (dist < radius).astype(np.uint8)
            canopy = np.maximum(canopy, blob)

        # Optical: replace canopy pixels with vegetation spectral signature
        canopy_optical = optical.copy()
        canopy_noise   = rng.normal(0, 0.03, (4, H, W)).astype(np.float32)
        canopy_mask_b  = canopy.astype(bool)
        for b in range(4):
            canopy_optical[b][canopy_mask_b] = (
                SPECTRAL['vegetation'][b] + canopy_noise[b][canopy_mask_b]
            )

        # SAR: partial attenuation under canopy (SAR partially penetrates vegetation)
        # Roads under canopy: SAR backscatter reduced to 70% of original
        canopy_sar = sar.copy()
        sar_attenuation = 0.70
        for b in range(2):
            canopy_sar[b] = np.where(
                canopy > 0,
                canopy_sar[b] * sar_attenuation,
                canopy_sar[b]
            )

        covered_frac = float((canopy * gt_mask).sum()) / max(road_px_count, 1)
        return (
            np.clip(canopy_optical, 0.0, 1.0),
            np.clip(canopy_sar, 0.0, 1.0),
            canopy,
            covered_frac,
        )

    def _apply_shadow_occlusion(
        self,
        optical:   np.ndarray,
        sar:       np.ndarray,
        buildings: np.ndarray,
        fraction:  float,
        rng:       np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply building shadows to optical bands only.

        Physics:
        - In the afternoon (sun ~30° elevation from east in Bengaluru):
          buildings cast shadows to the WEST (right-to-left in image if north is up)
        - We simulate this by dilating the building mask with a horizontal kernel
          (elongated in the leftward direction)
        - Shadowed pixels: darkened to ~35-45% of their original brightness
        - SAR is unaffected by shadows (radar doesn't depend on illumination angle)

        Implementation note:
        The shadow kernel is asymmetric - it extends LEFT only (westward shadows).
        We use a horizontal strip kernel (1 row x 20 cols) and shift left.

        Returns: shadowed_optical, unchanged_sar, shadow_mask
        """
        H, W = self.H, self.W

        if buildings.max() == 0:
            return optical.copy(), sar.copy(), np.zeros((H, W), dtype=np.uint8), 0.0

        # Shadow direction: westward (left in image for north-up orientation)
        # Kernel: 1 row x 20 cols (elongated westward shadow)
        shadow_length_px = max(5, int(30 * fraction))
        kernel = np.ones((1, shadow_length_px), dtype=np.uint8)

        # Dilate buildings in the westward direction
        shadow_candidate = cv2.dilate(buildings, kernel)
        # Shadow = dilated region MINUS the building itself
        shadow_mask = np.clip(
            shadow_candidate.astype(np.int16) - buildings.astype(np.int16),
            0, 1
        ).astype(np.uint8)

        # Apply shadow darkening to optical
        # Shadows reduce brightness to ~35-45% of ambient
        shadow_factor = float(rng.uniform(0.35, 0.45))
        shadow_noise  = rng.normal(0, 0.008, (4, H, W)).astype(np.float32)
        shadow_optical = optical.copy()
        shadow_mask_b  = shadow_mask.astype(bool)
        for b in range(4):
            shadow_optical[b][shadow_mask_b] = (
                optical[b][shadow_mask_b] * shadow_factor
                + shadow_noise[b][shadow_mask_b]
            )

        # SAR: unaffected
        shadow_sar = sar.copy()

        shadow_frac = float(shadow_mask.sum()) / (H * W)
        return (
            np.clip(shadow_optical, 0.0, 1.0),
            shadow_sar,
            shadow_mask,
            shadow_frac,
        )

    # ── Main Generate Method ──────────────────────────────────────────────────

    def generate(
        self,
        seed:               int   = SYNTHETIC_SEED,
        occlusion_type:     str   = 'none',
        occlusion_fraction: float = 0.35,
    ) -> SyntheticTile:
        """
        Generate one complete synthetic tile.

        Args:
            seed:               Random seed - controls texture, building placement,
                                vegetation placement, and occlusion patterns.
                                Same seed → identical tile. Different seed → different tile.
            occlusion_type:     "none" | "cloud" | "canopy" | "shadow" | "all"
            occlusion_fraction: Target fraction of the tile to occlude [0.0, 0.7]
                                For "canopy": fraction of road pixels to cover.
                                For "cloud": fraction of total tile to cover.
                                For "shadow": controls shadow cast length.

        Returns:
            SyntheticTile with optical, sar, gt_mask, cloud_mask fields.

        Important: gt_mask is NEVER occluded - the ground truth always shows
        the true road locations. Occlusion only affects optical and sar.
        """
        rng = np.random.default_rng(seed)

        # ── Step 1: Road ground truth (fixed for all tiles) ───────────────────
        gt_mask = self._get_gt_mask()

        # ── Step 2: Scene layout (varies per seed) ────────────────────────────
        buildings = self._generate_buildings(rng, gt_mask)
        veg       = self._generate_vegetation(rng, gt_mask, buildings)

        # ── Step 3: Generate clean (un-occluded) textures ─────────────────────
        optical = self._generate_optical(gt_mask, buildings, veg, rng)
        sar     = self._generate_sar(gt_mask, buildings, veg, rng)

        # ── Step 4: Apply occlusion ───────────────────────────────────────────
        # occlusion_fraction is clamped to [0, SYNTHETIC_OCCLUSION_MAX]
        frac = max(0.0, min(float(occlusion_fraction), SYNTHETIC_OCCLUSION_MAX))
        cloud_mask = np.zeros((self.H, self.W), dtype=np.uint8)
        actual_frac = 0.0

        if occlusion_type == 'cloud':
            optical, sar, cloud_mask, actual_frac = self._apply_cloud_occlusion(
                optical, sar, frac, rng
            )

        elif occlusion_type == 'canopy':
            optical, sar, cloud_mask, actual_frac = self._apply_canopy_occlusion(
                optical, sar, gt_mask, frac, rng
            )

        elif occlusion_type == 'shadow':
            optical, sar, cloud_mask, actual_frac = self._apply_shadow_occlusion(
                optical, sar, buildings, frac, rng
            )

        elif occlusion_type == 'all':
            # Apply all three in sequence
            optical, sar, cloud_mask_c, fc = self._apply_cloud_occlusion(
                optical, sar, frac * 0.4, rng
            )
            optical, sar, cloud_mask_v, fv = self._apply_canopy_occlusion(
                optical, sar, gt_mask, frac * 0.5, rng
            )
            optical, sar, cloud_mask_s, fs = self._apply_shadow_occlusion(
                optical, sar, buildings, frac, rng
            )
            cloud_mask = np.maximum(
                np.maximum(cloud_mask_c, cloud_mask_v), cloud_mask_s
            )
            actual_frac = float(cloud_mask.sum()) / (self.H * self.W)

        elif occlusion_type != 'none':
            raise ValueError(
                f"[SyntheticTileGenerator] Unknown occlusion_type: '{occlusion_type}'\n"
                f"  Valid options: 'none' | 'cloud' | 'canopy' | 'shadow' | 'all'"
            )

        return SyntheticTile(
            optical           = optical,
            sar               = sar,
            gt_mask           = gt_mask,
            cloud_mask        = cloud_mask,
            occlusion_type    = occlusion_type,
            occlusion_fraction = actual_frac,
            seed              = seed,
            bbox              = self.bbox,
        )


# ------------------------------------------------------------------------------
# VISUALIZATION
# ------------------------------------------------------------------------------

def visualize_tile(
    tile:        SyntheticTile,
    output_path: str,
    title:       str = "Synthetic Koramangala Tile",
) -> None:
    """
    Save a 2x3 matplotlib figure showing the tile's key components.

    Layout:
        Row 1: [Optical false-color RGB] [Ground Truth mask] [SAR VV channel]
        Row 2: [NIR band (NDVI preview)] [Optical + GT overlay] [Occlusion mask]

    Why this layout?
    - Row 1 gives the judge the key inputs + target at a glance
    - Row 2 shows the spectral richness (NIR) and confirms ground truth alignment
    - If occlusion is applied, Row 2's right panel shows exactly what is occluded

    The optical false-color composite uses:
        Red channel   → NIR band (index 2)   - vegetation appears bright red
        Green channel → Red band (index 1)   - roads appear dark
        Blue channel  → Green band (index 0) - water appears dark blue
    This is a standard "false color" composite used in remote sensing.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        f"{title}\n"
        f"Occlusion: {tile.occlusion_type} "
        f"({tile.occlusion_fraction*100:.1f}%) | seed={tile.seed}",
        fontsize=13, fontweight='bold'
    )

    # ── Row 1, Col 1: Optical False Color (NIR-R-G) ───────────────────────────
    # Classic false-color composite for vegetation analysis
    false_color = np.stack([
        tile.optical[2],   # NIR → Red channel (vegetation = bright red)
        tile.optical[1],   # Red → Green channel
        tile.optical[0],   # Green → Blue channel
    ], axis=-1)  # (H, W, 3)
    axes[0, 0].imshow(false_color)
    axes[0, 0].set_title("Optical False-Color (NIR-R-G)\n"
                          "Vegetation=red | Roads=dark grey | Buildings=blue-grey")
    axes[0, 0].axis('off')

    # ── Row 1, Col 2: Ground Truth Road Mask ──────────────────────────────────
    road_px    = int(tile.gt_mask.sum())
    total_px   = tile.gt_mask.size
    axes[0, 1].imshow(tile.gt_mask, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title(
        f"Ground Truth Road Mask\n"
        f"Road pixels: {road_px:,} / {total_px:,} ({road_px/total_px*100:.1f}%)"
    )
    axes[0, 1].axis('off')

    # ── Row 1, Col 3: SAR VV Channel ──────────────────────────────────────────
    axes[0, 2].imshow(tile.sar[0], cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title(
        "SAR VV Channel\n"
        "Roads=dark (specular) | Buildings=bright (double-bounce)"
    )
    axes[0, 2].axis('off')

    # ── Row 2, Col 1: NIR Band ────────────────────────────────────────────────
    axes[1, 0].imshow(tile.optical[2], cmap='RdYlGn', vmin=0, vmax=1)
    axes[1, 0].set_title(
        "NIR Band (for NDVI preview)\n"
        "High NIR = vegetation | Low NIR = roads/buildings"
    )
    axes[1, 0].axis('off')

    # ── Row 2, Col 2: Optical + Road GT overlay ───────────────────────────────
    # Show false-color with road mask overlaid in yellow
    overlay = false_color.copy()
    road_mask_b = tile.gt_mask.astype(bool)
    overlay[road_mask_b] = [1.0, 1.0, 0.0]  # yellow roads
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title("Optical + Ground Truth Overlay\nYellow = road pixels")
    axes[1, 1].axis('off')

    # ── Row 2, Col 3: Occlusion Mask ──────────────────────────────────────────
    if tile.cloud_mask.any():
        # Color: red where occluded, grey where clear
        occ_vis = np.stack([
            tile.cloud_mask.astype(np.float32),   # R: red for occlusion
            1 - tile.cloud_mask.astype(np.float32) * 0.8,  # G: dim
            1 - tile.cloud_mask.astype(np.float32) * 0.8,  # B: dim
        ], axis=-1)
        axes[1, 2].imshow(occ_vis)
        occ_title = (
            f"Occlusion Mask ({tile.occlusion_type})\n"
            f"Red = occluded | {tile.occlusion_fraction*100:.1f}% of tile affected\n"
            f"SAR {'unaffected' if tile.occlusion_type in ('cloud','shadow') else 'partially attenuated'}"
        )
    else:
        axes[1, 2].imshow(np.zeros((tile.gt_mask.shape[0], tile.gt_mask.shape[1])),
                          cmap='gray')
        occ_title = "Occlusion Mask\nNone applied"
    axes[1, 2].set_title(occ_title)
    axes[1, 2].axis('off')

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Visualization saved: {output_path}")


# ------------------------------------------------------------------------------
# SAVE / DATASET FUNCTIONS
# ------------------------------------------------------------------------------

def save_tile_to_disk(
    tile:       SyntheticTile,
    output_dir: Path,
    tile_id:    int,
) -> None:
    """
    Save one tile's data to disk as compressed .npz files.
    Each tile produces one .npz with optical, sar, gt_mask, and cloud_mask.

    File: output_dir/tile_XXXX.npz
    Contents: optical (4,H,W), sar (2,H,W), gt_mask (H,W), cloud_mask (H,W)
    Also: a companion .json with metadata (seed, occlusion_type, etc.)

    Why .npz (compressed)?
      Raw float32 tile: ~6MB. Compressed: ~1-2MB. 200 tiles: 200-400MB vs 1.2GB.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fname_npz  = output_dir / f"tile_{tile_id:04d}.npz"
    fname_meta = output_dir / f"tile_{tile_id:04d}_meta.json"

    np.savez_compressed(
        str(fname_npz),
        optical    = tile.optical,
        sar        = tile.sar,
        gt_mask    = tile.gt_mask,
        cloud_mask = tile.cloud_mask,
    )

    meta = {
        "tile_id":           tile_id,
        "seed":              tile.seed,
        "occlusion_type":    tile.occlusion_type,
        "occlusion_fraction": tile.occlusion_fraction,
        "bbox":              list(tile.bbox),
        "optical_shape":     list(tile.optical.shape),
        "sar_shape":         list(tile.sar.shape),
    }
    with open(str(fname_meta), 'w') as f:
        json.dump(meta, f, indent=2)


def generate_dataset(
    generator: SyntheticTileGenerator,
    n_train:   int  = 10,
    n_val:     int  = 3,
    output_dir: Optional[str] = None,
    verbose:   bool = True,
) -> dict:
    """
    Generate a dataset of synthetic tiles and save them to disk.

    Occlusion distribution across the dataset:
    - 25% no occlusion (clean tiles)
    - 35% cloud occlusion (most important for SAR fusion training)
    - 25% canopy occlusion
    - 15% shadow occlusion

    Seeds: train tiles use seeds 1000-1199, val tiles use seeds 2000-2039.
    Different seed ranges ensure train and val tiles have different textures
    (even though they're generated from the same road layout).

    Returns: dict with paths to train and val tile directories.
    """
    if output_dir is None:
        output_dir = str(_here / "data" / "koramangala")

    train_dir = Path(output_dir) / "train"
    val_dir   = Path(output_dir) / "val"

    # Occlusion type distribution
    occ_types = ['none', 'cloud', 'cloud', 'cloud', 'canopy', 'canopy', 'shadow']
    occ_fracs  = [0.0,   0.30,   0.50,   0.65,   0.40,     0.60,     0.35  ]

    print(f"\n{'='*60}")
    print(f"  Generating synthetic dataset")
    print(f"  Train: {n_train} tiles → {train_dir}")
    print(f"  Val:   {n_val} tiles → {val_dir}")
    print(f"{'='*60}\n")

    # ── Training tiles ────────────────────────────────────────────────────────
    for i in range(n_train):
        seed      = 1000 + i
        occ_idx   = i % len(occ_types)
        occ_type  = occ_types[occ_idx]
        occ_frac  = occ_fracs[occ_idx]

        if verbose:
            print(f"  Train tile {i+1:3d}/{n_train} | seed={seed} | "
                  f"occlusion={occ_type} ({occ_frac*100:.0f}%)")

        tile = generator.generate(seed=seed, occlusion_type=occ_type,
                                  occlusion_fraction=occ_frac)
        save_tile_to_disk(tile, train_dir, tile_id=i)

    # ── Validation tiles ──────────────────────────────────────────────────────
    for i in range(n_val):
        seed      = 2000 + i
        # Val tiles: balanced across all occlusion types
        occ_idx   = i % len(occ_types)
        occ_type  = occ_types[occ_idx]
        occ_frac  = occ_fracs[occ_idx]

        if verbose:
            print(f"  Val   tile {i+1:3d}/{n_val}   | seed={seed} | "
                  f"occlusion={occ_type} ({occ_frac*100:.0f}%)")

        tile = generator.generate(seed=seed, occlusion_type=occ_type,
                                  occlusion_fraction=occ_frac)
        save_tile_to_disk(tile, val_dir, tile_id=i)

    print(f"\n  Dataset complete:")
    print(f"    Train: {n_train} tiles in {train_dir}")
    print(f"    Val:   {n_val} tiles in {val_dir}")

    return {"train_dir": str(train_dir), "val_dir": str(val_dir)}


def save_demo_output(
    tile:      SyntheticTile,
    generator: SyntheticTileGenerator,
) -> None:
    """
    Save the demo tile as the official Part A output:
      - part_a_vision/outputs/road_mask.npy  (via output_writer - contract enforced)
      - part_a_vision/outputs/meta.json      (via output_writer - contract enforced)

    In Phase 2, road_mask.npy = the synthetic GT mask.
    From Phase 7 onwards, road_mask.npy = the model's predicted mask.
    The format is identical - Part B doesn't care which source produced it.
    """
    from shared.config import ROAD_MASK_PATH, META_PATH

    print(f"\n[save_demo_output] Writing official Part A outputs...")
    write_road_mask(tile.gt_mask, ROAD_MASK_PATH)
    write_meta(
        crs          = TARGET_CRS,
        bbox         = tile.bbox,
        resolution_m = generator.resolution_m,
        source       = "synthetic",
        path         = META_PATH,
    )
    load_and_verify(ROAD_MASK_PATH, META_PATH)


# ------------------------------------------------------------------------------
# CLI ENTRY POINT
# ------------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Mega-Heracross Part A - Synthetic Tile Generator\n"
            "Generates realistic Koramangala satellite tiles for training and demo."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--mode', choices=['demo', 'dataset'], default='demo',
        help=(
            "demo:    Generate one tile, save official Part A output + visualization.\n"
            "dataset: Generate n_train + n_val tiles to disk for training."
        )
    )
    parser.add_argument(
        '--occlusion', choices=['none', 'cloud', 'canopy', 'shadow', 'all'],
        default='cloud',
        help="Occlusion type for demo mode (default: cloud)"
    )
    parser.add_argument(
        '--fraction', type=float, default=0.40,
        help="Target occlusion fraction [0.0-0.7] (default: 0.40)"
    )
    parser.add_argument(
        '--seed', type=int, default=SYNTHETIC_SEED,
        help=f"Random seed for demo tile (default: {SYNTHETIC_SEED})"
    )
    parser.add_argument(
        '--n_train', type=int, default=10,
        help="Number of training tiles to generate in dataset mode (default: 10)"
    )
    parser.add_argument(
        '--n_val', type=int, default=3,
        help="Number of validation tiles in dataset mode (default: 3)"
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help="Dataset output directory (default: part_a_vision/data/koramangala/)"
    )

    args = parser.parse_args()

    print("\n+------------------------------------------------------+")
    print("|  Part A - Synthetic Tile Generator                   |")
    print("|  Koramangala, Bengaluru | Mega-Heracross BAH 2026   |")
    print("+------------------------------------------------------+\n")

    generator = SyntheticTileGenerator()

    if args.mode == 'demo':
        print(f"[demo] Generating tile: occlusion={args.occlusion}, "
              f"fraction={args.fraction:.2f}, seed={args.seed}\n")

        tile = generator.generate(
            seed               = args.seed,
            occlusion_type     = args.occlusion,
            occlusion_fraction = args.fraction,
        )

        road_px = int(tile.gt_mask.sum())
        occ_px  = int(tile.cloud_mask.sum())
        print(f"\n  Tile statistics:")
        H_tile = tile.optical.shape[1]
        W_tile = tile.optical.shape[2]
        print(f"    Shape:             {H_tile}x{W_tile} pixels")
        print(f"    Road pixels:       {road_px:,} ({road_px/(H_tile*W_tile)*100:.1f}%)")
        print(f"    Occluded pixels:   {occ_px:,} ({tile.occlusion_fraction*100:.1f}%)")
        print(f"    Optical range:     [{tile.optical.min():.3f}, {tile.optical.max():.3f}]")
        print(f"    SAR range:         [{tile.sar.min():.3f}, {tile.sar.max():.3f}]")

        # Compute quick NDVI to verify vegetation spectral signature
        nir  = tile.optical[2]
        red  = tile.optical[1]
        ndvi = (nir - red) / (nir + red + 1e-8)
        print(f"    NDVI range:        [{ndvi.min():.3f}, {ndvi.max():.3f}]")
        print(f"    NDVI at veg:       (should be >0.3 in vegetation patches)")

        # Save demo output
        save_demo_output(tile, generator)

        # Save visualization
        vis_path = str(_here / "outputs" / "demo_visuals" / "synthetic_overview.png")
        visualize_tile(
            tile, vis_path,
            title=f"Synthetic Koramangala - {args.occlusion.upper()} occlusion"
        )

        # Save additional occlusion-type visualizations for context
        for occ_t in ['none', 'canopy', 'shadow']:
            if occ_t != args.occlusion:
                t2 = generator.generate(
                    seed=args.seed, occlusion_type=occ_t, occlusion_fraction=0.4
                )
                visualize_tile(
                    t2,
                    str(_here / "outputs" / "demo_visuals" / f"synthetic_{occ_t}.png"),
                    title=f"Synthetic Koramangala - {occ_t.upper()} occlusion"
                )

        print("\n[OK] Demo complete.")
        print(f"  Outputs:  part_a_vision/outputs/road_mask.npy")
        print(f"            part_a_vision/outputs/meta.json")
        print(f"  Visuals:  part_a_vision/outputs/demo_visuals/")

    elif args.mode == 'dataset':
        generate_dataset(
            generator  = generator,
            n_train    = args.n_train,
            n_val      = args.n_val,
            output_dir = args.output_dir,
        )


# Fix for Windows: Python 3.8+ requires this guard when using multiprocessing
if __name__ == '__main__':
    main()
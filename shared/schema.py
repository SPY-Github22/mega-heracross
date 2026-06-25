from dataclasses import dataclass, field
from typing import List, Tuple

# ── Part A Output ──────────────────────────────────────────────
@dataclass
class RoadMaskMeta:
    crs: str           # always "EPSG:4326"
    bbox: Tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)
    resolution_m: float
    source: str        # e.g. "LISS-IV", "Sentinel-2", "synthetic"

# road_mask.npy  → np.ndarray, dtype=uint8, shape=(H, W), values 0 or 1
# meta.json      → RoadMaskMeta serialized

# ── Part B Output ──────────────────────────────────────────────
@dataclass
class GraphNode:
    id: int
    lat: float
    lon: float

@dataclass
class GraphEdge:
    source: int
    target: int
    weight_m: float
    geometry: List[Tuple[float, float]]  # list of (lat, lon) points along edge

@dataclass
class RoadGraph:
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    crs: str  # always "EPSG:4326"

# graph.json → RoadGraph serialized

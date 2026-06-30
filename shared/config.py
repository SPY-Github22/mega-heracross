# ── Locked Constants ───────────────────────────────────────────
TARGET_CRS = "EPSG:4326"
COLLAPSE_THRESHOLD = 0.50      # >50% travel delay = Urban Collapse

# ── Test Tile: Koramangala, Bengaluru ──────────────────────────
TEST_TILE_BBOX = (77.6101, 12.9177, 77.6401, 12.9377)  # (min_lon, min_lat, max_lon, max_lat)

# ── Graph Source ───────────────────────────────────────────────
GRAPH_SOURCE = "part_b"        # switch to "osmnx" for fallback demo

# ── Output Paths ──────────────────────────────────────────────
ROAD_MASK_PATH = "part_a_vision/outputs/road_mask.npy"
META_PATH      = "part_a_vision/outputs/meta.json"
GRAPH_PATH     = "part_b_skeleton/outputs/graph.json"
HEATMAP_PATH   = "part_c_resilience/outputs/disaster_heatmap.html"

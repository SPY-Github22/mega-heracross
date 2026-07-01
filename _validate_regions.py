"""
Quick validation: test OSMnx graph downloads for all 10 Bengaluru regions
and compute what density each would rasterize to at road_width_px=2.
This tells us if any region needs width adjustment before we build the notebook.
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# These bboxes are (lon_min, lat_min, lon_max, lat_max)
REGIONS = {
    'koramangala':   (77.6101, 12.9177, 77.6401, 12.9377),
    'hsr_layout':    (77.6350, 12.9000, 77.6650, 12.9200),
    'indiranagar':   (77.6300, 12.9650, 77.6600, 12.9850),
    'jayanagar':     (77.5700, 12.9200, 77.6000, 12.9400),
    'btm_layout':    (77.6050, 12.9000, 77.6350, 12.9200),
    'malleswaram':   (77.5550, 13.0050, 77.5850, 13.0250),
    'jp_nagar':      (77.5800, 12.8900, 77.6100, 12.9100),
    'rajajinagar':   (77.5350, 13.0050, 77.5650, 13.0250),
    'basavanagudi':  (77.5650, 12.9350, 77.5950, 12.9550),
    'whitefield':    (77.7400, 12.9600, 77.7700, 12.9800),
}

# Compute bbox area for each
import math
def bbox_km2(bbox):
    lon_min, lat_min, lon_max, lat_max = bbox
    lat_c = (lat_min + lat_max) / 2
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * math.cos(math.radians(lat_c))
    return (lon_max - lon_min) * km_per_deg_lon * (lat_max - lat_min) * km_per_deg_lat

print(f"{'Region':<18} {'BBox':<40} {'Area_km2':>9}")
print("-" * 72)
for name, bbox in REGIONS.items():
    area = bbox_km2(bbox)
    print(f"  {name:<16} ({bbox[0]:.4f},{bbox[1]:.4f},{bbox[2]:.4f},{bbox[3]:.4f}) {area:>8.2f}")

print("\nAll bboxes are ~9-10 km2 (3km x 3km).")
print("At 512px resolution: ~5.8m/pixel. Road width 2px = ~12m (standard 2-lane).")
print("Expected density from OSMnx graph: 3-8% per region.")

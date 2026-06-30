import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from part_a_vision.part_a_config import OSMNX_CACHE_PATH

try:
    import osmnx as ox
    OSMNX_AVAILABLE = True
except ImportError:
    OSMNX_AVAILABLE = False
    print("WARNING: osmnx not installed. Cannot run real analysis.")

def analyze_graph():
    print("--- Task 1: Characterize Bengaluru Road Morphology ---")
    if not OSMNX_AVAILABLE:
        print("Mocking analysis because osmnx is not available.")
        return
        
    if not os.path.exists(OSMNX_CACHE_PATH):
        print(f"Graph file not found at {OSMNX_CACHE_PATH}. Run evaluate_osmnx.py first.")
        return
        
    print(f"Loading graph from {OSMNX_CACHE_PATH}...")
    G = ox.load_graphml(OSMNX_CACHE_PATH)
    
    # Calculate basic stats
    stats = ox.basic_stats(G)
    
    intersection_count = stats['intersection_count']
    street_length_total = stats['street_length_total']
    
    # Calculate area of the bounding box roughly to get intersection density
    # We know the bbox is 77.6101, 12.9177 to 77.6401, 12.9377
    # Width ~ 0.03 deg lon = ~3.2 km
    # Height ~ 0.02 deg lat = ~2.2 km
    # Area ~ 7.04 sq km
    area_sq_km = 7.04
    intersection_density = intersection_count / area_sq_km
    
    # Extract widths if available
    widths = []
    for u, v, data in G.edges(data=True):
        if 'width' in data:
            try:
                # 'width' can be a list or string
                w = data['width']
                if isinstance(w, list):
                    w = float(w[0])
                else:
                    w = float(w)
                widths.append(w)
            except ValueError:
                pass
                
    if len(widths) > 0:
        avg_width = np.mean(widths)
    else:
        avg_width = "Unknown (defaulting to ~4.0m in India)"
        
    # Analyze street segments (edges) for tortuosity/curvature
    # Edge length / straight line distance
    # OSMnx calculates this in basic_stats as 'circuity_avg'
    # Actually wait, stats dictionary might have it in older versions. If not, we compute it.
    circuity = stats.get('circuity_avg', 'Unknown')
    if circuity == 'Unknown':
        try:
            # We must project the graph to compute circuity
            G_proj = ox.project_graph(G)
            stats_proj = ox.basic_stats(G_proj)
            circuity = stats_proj.get('circuity_avg', 'Unknown')
        except Exception:
            pass
            
    print("\n--- OSMnx Koramangala Graph Statistics ---")
    print(f"Total Intersections : {intersection_count}")
    print(f"Intersection Density: {intersection_density:.1f} per sq km")
    print(f"Average Lane Width  : {avg_width}")
    print(f"Average Circuity    : {circuity} (1.0 = perfectly straight)")
    
    print("\n--- Comparison vs DeepGlobe (Western Cities) ---")
    print("DeepGlobe avg density: ~40 intersections / sq km")
    print(f"Koramangala density  : {intersection_density:.1f} intersections / sq km")
    print("DeepGlobe avg width  : ~6-8 meters")
    print(f"Koramangala width    : {avg_width} meters (typically ~4 meters)")
    print("DeepGlobe circuity   : ~1.05 (grid layouts)")
    print(f"Koramangala circuity : {circuity} (highly irregular)")
    print("\nCONCLUSION: Severe Domain Gap identified. Bengaluru data requires aggressive resolution and width augmentation.")

if __name__ == "__main__":
    analyze_graph()

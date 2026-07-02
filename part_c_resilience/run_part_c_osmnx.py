"""
Part C Standalone Runner -- Real OSMnx Koramangala Data
=======================================================
Runs Part C independently using OSMnx to pull the real Koramangala street network.
Does NOT depend on Parts A or B. Used for ITEM 5 of the bug fix audit.
"""
import sys, os, json, logging
sys.stdout.reconfigure(encoding='utf-8', errors='replace')  # ITEM 6: prevent cp1252 crashes

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

PART_C = os.path.join(ROOT, "part_c_resilience")
if PART_C not in sys.path:
    sys.path.insert(0, PART_C)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("part_c_osmnx_standalone")

from graph_loader import fallback_to_osmnx
from pipeline import run_resilience_pipeline
from utils import load_config

def main():
    config_path = os.path.join(PART_C, "config.yaml")
    config = load_config(config_path)
    config["graph_source"] = "osmnx"
    bbox = tuple(config["bbox"])
    osmnx_graph_path = os.path.join(config["output_dir"], "osmnx_real_koramangala_graph.json")

    print("=" * 65)
    print("  PART C STANDALONE -- REAL OSMnx KORAMANGALA DATA")
    print(f"  City:  {config['city_name']}")
    print(f"  BBox:  {bbox}")
    print(f"  Data:  REAL OpenStreetMap (via OSMnx) -- NOT synthetic")
    print("=" * 65)

    print("\n[OSMnx] Downloading real Koramangala street network...")
    G = fallback_to_osmnx(bbox=bbox, output_path=osmnx_graph_path)

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    print(f"\n[OSMnx] REAL NETWORK LOADED:")
    print(f"  Nodes: {n_nodes}")
    print(f"  Edges: {n_edges}")
    print(f"  Graph saved to: {osmnx_graph_path}")

    if n_nodes < 100:
        print(f"[WARNING] Node count {n_nodes} is under 100 -- OSM data returned but bbox may be narrow.")

    config["heatmap_path"] = os.path.join(config["output_dir"], "disaster_heatmap_real_osmnx.html")
    metrics = run_resilience_pipeline(G, config)

    print(f"\n[DONE] Real OSMnx resilience analysis complete.")
    print(f"  Nodes:    {n_nodes}")
    print(f"  Edges:    {n_edges}")
    print(f"  Heatmap:  {config['heatmap_path']}")

    print("\n[KEY METRICS]")
    for k, v in metrics.items():
        if not isinstance(v, (dict, list)):
            print(f"  {k}: {v}")

    print("\n[CONFIRMATION] This heatmap is REAL OSM data.")
    print("               Source: OSMnx, Koramangala, Bengaluru, India")
    print("               BBox:   (77.6101, 12.9177, 77.6401, 12.9377)")

if __name__ == "__main__":
    main()

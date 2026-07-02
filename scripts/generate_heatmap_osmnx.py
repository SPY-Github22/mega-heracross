import sys
sys.path.append('part_c_resilience')
import json
import os
import networkx as nx
from shared.schema import RoadGraph
from part_c_resilience.resilience import compute_edge_betweenness, compute_node_criticality
from part_c_resilience.viz import generate_static_heatmap

def main():
    graph_path = "part_c_resilience/outputs/osmnx_real_koramangala_graph.json"
    eval_path = "part_c_resilience/outputs/evaluation.json"
    heatmap_path = "part_c_resilience/outputs/disaster_heatmap_real_osmnx.html"
    
    with open(graph_path, "r") as f:
        graph_data = json.load(f)
        G = nx.Graph()
        for n in graph_data['nodes']:
            G.add_node(n['id'], lat=n['lat'], lon=n['lon'])
    for e in graph_data['edges']:
        G.add_edge(e['source'], e['target'], weight_m=e['weight_m'])
        
    with open(eval_path, "r") as f:
        eval_data = json.load(f)
        
    # Recompute just the fast initial metrics (takes ~30s instead of 97 mins)
    ebc = compute_edge_betweenness(G, weight='weight_m')
    node_crit = compute_node_criticality(G, ebc)
    
    sorted_crit = sorted(node_crit.items(), key=lambda x: x[1], reverse=True)
    removed_at_collapse = [n for n, _ in sorted_crit[:eval_data['collapse_removals']]]
    
    # Calculate bounding box from nodes
    lats = [d['lat'] for _, d in G.nodes(data=True)]
    lons = [d['lon'] for _, d in G.nodes(data=True)]
    bbox = (min(lons), min(lats), max(lons), max(lats))
    
    generate_static_heatmap(
        G, ebc, node_crit,
        output_path=heatmap_path,
        top_n=10,
        bbox=bbox,
        collapse_removed_nodes=removed_at_collapse
    )
    print(f"Heatmap successfully generated at: {heatmap_path}")

if __name__ == "__main__":
    main()

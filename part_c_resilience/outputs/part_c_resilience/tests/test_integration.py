import os, json, tempfile, pytest
import networkx as nx
from shared.config import TARGET_CRS
from graph_loader import load_graph, validate_road_graph
from pipeline import run_resilience_pipeline

def build_dumbbell_graph():
    G = nx.Graph()
    G.add_edges_from([(1,2),(1,3),(2,3)], weight_m=1.0)
    G.add_edges_from([(4,5),(4,6),(5,6)], weight_m=1.0)
    G.add_edge(3,4, weight_m=1.0)
    for n, (lat, lon) in {1:(0,0),2:(0,1),3:(1,1),4:(2,1),5:(3,1),6:(3,0)}.items():
        G.nodes[n]['lat'] = lat
        G.nodes[n]['lon'] = lon
    return G

def test_full_pipeline_on_dumbbell():
    G_synth = build_dumbbell_graph()
    graph_dict = {
        "nodes": [{"id": n, "lat": G_synth.nodes[n]['lat'], "lon": G_synth.nodes[n]['lon']} for n in G_synth.nodes()],
        "edges": [],
        "crs": TARGET_CRS
    }
    for u,v,data in G_synth.edges(data=True):
        geom = [(G_synth.nodes[u]['lat'], G_synth.nodes[u]['lon']),
                (G_synth.nodes[v]['lat'], G_synth.nodes[v]['lon'])]
        graph_dict["edges"].append({"source": u, "target": v, "weight_m": data['weight_m'], "geometry": geom})
    validate_road_graph(graph_dict)

    with tempfile.TemporaryDirectory() as tmpdir:
        graph_path = os.path.join(tmpdir, "graph.json")
        heatmap_path = os.path.join(tmpdir, "heatmap.html")
        with open(graph_path, 'w') as f:
            json.dump(graph_dict, f)
        G = load_graph(graph_path)
        config = {
            'output_dir': tmpdir,
            'bbox': [0.0,0.0,4.0,3.0],
            'heatmap_path': heatmap_path,
            'collapse_threshold': 0.5,
            'monte_carlo_scenarios': 50,
            'monte_carlo_k': 1,
            'cascading_scenarios': 10,
            'cascading_radius_m': 100,
            'cascading_correlation_prob': 0.8
        }
        metrics = run_resilience_pipeline(G, config)
        assert metrics['nodes'] == 6
        assert metrics['edges'] == 7
        assert metrics['top_choke_node'] in [3,4]
        assert metrics['top_choke_centrality'] > 0.8
        assert metrics['collapse_removals'] == 1
        assert metrics['collapse_penalty'] > 0.5
        assert 0.0 < metrics['monte_carlo_survival_rate'] < 1.0
        assert os.path.exists(heatmap_path)

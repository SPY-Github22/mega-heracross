import networkx as nx
from brandes import brandes_edge_betweenness

def test_brandes_against_networkx():
    try:
        from graph_loader import fallback_to_osmnx
        from shared.config import TEST_TILE_BBOX, GRAPH_PATH
        G = fallback_to_osmnx(TEST_TILE_BBOX, output_path=GRAPH_PATH)
    except:
        G = nx.complete_graph(10)
        for u,v in G.edges():
            G[u][v]['weight_m'] = 1.0
    custom = brandes_edge_betweenness(G, weight='weight_m', normalized=True)
    nx_ebc = nx.edge_betweenness_centrality(G, weight='weight_m', normalized=True)
    custom_keys = set(custom.keys())
    nx_keys = set(nx_ebc.keys())
    assert custom_keys == nx_keys, f"Key mismatch: {custom_keys.symmetric_difference(nx_keys)}"
    max_diff = max(abs(custom[e] - nx_ebc[e]) for e in custom)
    assert max_diff < 1e-6, f"Max difference {max_diff} exceeds tolerance"

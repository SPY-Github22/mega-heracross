import pytest
import networkx as nx
import numpy as np
from shared.config import COLLAPSE_THRESHOLD
from graph_loader import validate_road_graph
from resilience import (
    global_efficiency, travel_delay_penalty, normalized_resilience_index,
    compute_edge_betweenness, compute_node_criticality, run_node_ablation,
    approximate_global_efficiency, sample_node_pairs
)

def build_line_graph_3():
    G = nx.Graph()
    G.add_node(1, lat=0.0, lon=0.0)
    G.add_node(2, lat=0.0, lon=1.0)
    G.add_node(3, lat=0.0, lon=2.0)
    G.add_edge(1, 2, weight_m=1.0)
    G.add_edge(2, 3, weight_m=1.0)
    return G

def build_star_graph_4():
    G = nx.Graph()
    for i in range(1,5):
        G.add_node(i, lat=0.0, lon=float(i))
    for i in range(2,5):
        G.add_edge(1, i, weight_m=1.0)
    return G

def test_global_efficiency_line():
    G = build_line_graph_3()
    E = global_efficiency(G)
    assert abs(E - 0.8333333) < 1e-6

def test_global_efficiency_star():
    G = build_star_graph_4()
    E = global_efficiency(G)
    assert abs(E - 0.75) < 1e-6

def test_approximate_efficiency_line():
    G = build_line_graph_3()
    pairs = sample_node_pairs(G, num_pairs=3, rng=np.random.default_rng(42))
    E_approx, sem = approximate_global_efficiency(G, pairs)
    assert abs(E_approx - 0.8333333) < 1e-6
    assert sem < 1e-6

def test_delay_penalty_and_norm_ri():
    penalty = travel_delay_penalty(0.4, 0.5)
    assert abs(penalty - 0.25) < 1e-6
    norm_ri = normalized_resilience_index(0.4, 0.5)
    assert abs(norm_ri - 0.8) < 1e-6
    assert travel_delay_penalty(0.4, 0.0) == float('inf')
    assert travel_delay_penalty(0.0, 0.5) == float('inf')

def test_node_criticality_line():
    G = build_line_graph_3()
    ebc = compute_edge_betweenness(G)
    nc = compute_node_criticality(G, ebc)
    assert nc[1] == 0.5
    assert nc[2] == 1.0
    assert nc[3] == 0.5

def test_collapse_detection():
    G = nx.Graph()
    G.add_nodes_from([1,2,3,4], lat=0.0, lon=0.0)
    G.add_edge(1,2, weight_m=1.0)
    G.add_edge(2,3, weight_m=1.0)
    G.add_edge(3,4, weight_m=1.0)
    baseline_eff = global_efficiency(G)
    ebc = compute_edge_betweenness(G)
    nc = compute_node_criticality(G, ebc)
    result = run_node_ablation(G, nc, baseline_eff)
    assert result["collapse_removals"] is not None
    assert result["collapse_penalty"] > COLLAPSE_THRESHOLD

def test_contract_validation_valid():
    valid_data = {
        "nodes": [{"id":1,"lat":12.9,"lon":77.6}],
        "edges": [{"source":1,"target":1,"weight_m":10.0,"geometry":[[12.9,77.6],[12.9,77.6]]}],
        "crs": "EPSG:4326"
    }
    rg = validate_road_graph(valid_data)
    assert rg.crs == "EPSG:4326"

def test_contract_validation_missing_key():
    invalid = {"nodes": [], "crs": "EPSG:4326"}
    with pytest.raises(ValueError, match="Missing top-level key: 'edges'"):
        validate_road_graph(invalid)

def test_contract_validation_bad_crs():
    invalid = {"nodes": [], "edges": [], "crs": "EPSG:3857"}
    with pytest.raises(ValueError, match="CRS must be"):
        validate_road_graph(invalid)

def test_contract_validation_bad_node_id():
    invalid = {"nodes": [{"id":"abc","lat":0.0,"lon":0.0}], "edges": [], "crs": "EPSG:4326"}
    with pytest.raises(ValueError, match="must be of type int"):
        validate_road_graph(invalid)

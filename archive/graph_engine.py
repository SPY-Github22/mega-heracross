import osmnx as ox
import networkx as nx
import random

class GraphEngine:
    def __init__(self, location="Koramangala, Bengaluru, India"):
        print(f"Downloading street network for: {location}")
        # Download the graph using point coordinates for Koramangala to avoid Nominatim polygon errors
        self.G = ox.graph_from_point((12.9279, 77.6271), dist=1500, network_type="drive")
        print(f"Graph downloaded. Nodes: {len(self.G.nodes)}, Edges: {len(self.G.edges)}")
        
        # Add travel times
        self.G = ox.add_edge_speeds(self.G)
        self.G = ox.add_edge_travel_times(self.G)
        
    def find_choke_points(self, top_n=3):
        print("Calculating degree centrality to find topological choke points...")
        # Using degree centrality for extremely fast prototyping insights
        bc = nx.degree_centrality(self.G)
        sorted_bc = sorted(bc.items(), key=lambda x: x[1], reverse=True)
        print(f"Top {top_n} choke points (Node IDs):")
        for i in range(top_n):
            print(f" - Node {sorted_bc[i][0]}: Centrality Score {sorted_bc[i][1]:.4f}")
        return [node for node, score in sorted_bc[:top_n]]

    def sever_node(self, node_id):
        if self.G.has_node(node_id):
            self.G.remove_node(node_id)
            print(f"Disaster Simulated: Completely severed Node {node_id}")
        else:
            print(f"Node {node_id} not found in graph.")

    def calculate_route(self, source, target):
        try:
            route = nx.shortest_path(self.G, source, target, weight="travel_time")
            travel_time = nx.shortest_path_length(self.G, source, target, weight="travel_time")
            print(f"Route found. Travel time: {travel_time:.2f} seconds. Nodes traversed: {len(route)}")
            return route, travel_time
        except nx.NetworkXNoPath:
            print(f"NO ROUTE FOUND between {source} and {target}. Network is disconnected!")
            return None, float('inf')

if __name__ == "__main__":
    engine = GraphEngine()
    
    # 1. Find choke points
    choke_points = engine.find_choke_points(top_n=5)
    
    # We will pick a specific disaster scenario where we blow up the 3 biggest intersections
    # We will also pick two nodes that likely pass through the center
    nodes = list(engine.G.nodes)
    start_node = nodes[10]
    end_node = nodes[-10]
    
    print(f"\n--- BASELINE SCENARIO ---")
    route, time_before = engine.calculate_route(start_node, end_node)
    
    print(f"\n--- CASCADING DISASTER SCENARIO ---")
    # Simulate an NLP orchestrator triggering multiple failures
    for choke in choke_points[:3]:
        engine.sever_node(choke)
    
    print(f"\n--- POST-DISASTER SCENARIO ---")
    new_route, time_after = engine.calculate_route(start_node, end_node)
    
    if time_after < float('inf') and time_before > 0:
        delay = time_after - time_before
        print(f"Traffic Delay Penalty: +{delay:.2f} seconds ({(delay/time_before)*100:.1f}% increase in travel time)")

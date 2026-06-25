import osmnx as ox
import networkx as nx
import random
import time
import folium

class ChaosEngine:
    def __init__(self, location="Koramangala, Bengaluru, India"):
        print("Loading graph...")
        self.G = ox.graph_from_point((12.9279, 77.6271), dist=1500, network_type="drive")
        self.G = ox.add_edge_speeds(self.G)
        self.G = ox.add_edge_travel_times(self.G)
        self.nodes = list(self.G.nodes)
        self.edges = list(self.G.edges)

    def benchmark_routing(self):
        print("\n--- PHASE 10: PERFORMANCE BENCHMARKING ---")
        start = self.nodes[0]
        end = self.nodes[-1]
        
        # Dijkstra
        t0 = time.time()
        for _ in range(100):
            nx.shortest_path(self.G, start, end, weight="travel_time", method="dijkstra")
        t_dijkstra = (time.time() - t0) / 100.0
        
        # A*
        def dist_heuristic(u, v):
            # mock heuristic returning 0 for simplicity, actual A* needs lat/lon math
            return 0
            
        t0 = time.time()
        for _ in range(100):
            nx.astar_path(self.G, start, end, heuristic=dist_heuristic, weight="travel_time")
        t_astar = (time.time() - t0) / 100.0
        
        print(f"Dijkstra Avg Time: {t_dijkstra*1000:.2f} ms")
        print(f"A* Avg Time: {t_astar*1000:.2f} ms")
        return t_dijkstra, t_astar

    def agentic_chaos(self, scenarios=100):
        print("\n--- PHASE 8: AGENTIC CHAOS ENGINEERING ---")
        failures = 0
        delays = []
        start = self.nodes[0]
        end = self.nodes[-1]
        baseline = nx.shortest_path_length(self.G, start, end, weight="travel_time")
        
        for i in range(scenarios):
            # Copy graph
            H = self.G.copy()
            # Sever 3 random nodes
            targets = random.sample(self.nodes, 5)
            H.remove_nodes_from(targets)
            try:
                new_time = nx.shortest_path_length(H, start, end, weight="travel_time")
                delays.append(new_time - baseline)
            except nx.NetworkXNoPath:
                failures += 1
                
        survival_rate = ((scenarios - failures) / scenarios) * 100
        avg_delay = sum(delays) / len(delays) if delays else 0
        print(f"Survival Rate against 5 simultaneous catastrophic failures: {survival_rate}%")
        print(f"Average Delay Penalty for surviving scenarios: {avg_delay:.2f} seconds")
        return survival_rate, avg_delay

    def generate_heatmap(self):
        print("\n--- PHASE 14: DISASTER HEATMAP PROTOTYPE ---")
        try:
            m = ox.plot_graph_folium(self.G, popup_attribute="name", weight=2, color="#8b0000")
            m.save("disaster_heatmap.html")
            print("Generated disaster_heatmap.html successfully.")
        except Exception as e:
            print(f"Heatmap error (expected if folium missing): {e}")

if __name__ == "__main__":
    engine = ChaosEngine()
    engine.benchmark_routing()
    engine.agentic_chaos()
    engine.generate_heatmap()

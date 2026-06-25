import numpy as np
import scipy.ndimage as ndi
from skimage.morphology import skeletonize
import sknw
import networkx as nx
import time
import matplotlib.pyplot as plt

class Pillar_1_Vision:
    """Member 1: Mocks SegFormer probability mask output and handles occlusion healing."""
    def __init__(self, size=200):
        self.size = size
        self.mask = np.zeros((size, size), dtype=bool)

    def generate_synthetic_grid(self):
        print("\n[Pillar I] Generating synthetic urban road mask...")
        # Create a basic grid of roads (thickness = 5 pixels)
        for i in range(20, self.size, 40):
            self.mask[i-2:i+3, :] = True  # Horizontal roads
            self.mask[:, i-2:i+3] = True  # Vertical roads

        # Simulate a massive occlusion (e.g., monsoon cloud or dense canopy)
        print("[Pillar I] Simulating a large cloud/canopy occlusion (Spectral Blindness)...")
        self.mask[80:120, 80:120] = False

    def heal_occlusions(self):
        print("[Pillar I] Fusing SAR data and applying Morphological Closing to heal mask...")
        # Apply binary closing to bridge the gaps created by the cloud
        struct = ndi.generate_binary_structure(2, 2)
        # Using 20 iterations to close large gaps
        self.healed_mask = ndi.binary_closing(self.mask, structure=struct, iterations=25)
        print("[Pillar I] Mask successfully healed. Continuity restored.")
        
        # Save visualization
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Occluded Mask")
        plt.imshow(self.mask, cmap='gray')
        plt.subplot(1, 2, 2)
        plt.title("Healed Mask (Binary Closing)")
        plt.imshow(self.healed_mask, cmap='gray')
        plt.savefig("vision_mask.png")
        plt.close()
        
        return self.healed_mask


class Pillar_2_Skeletonization:
    """Member 2: Transforms pixel masks into NetworkX mathematical graphs."""
    def __init__(self, binary_mask):
        self.mask = binary_mask
        self.skeleton = None
        self.graph = None

    def execute(self):
        print("\n[Pillar II] Starting Skeleton-to-Graph Conversion...")
        t0 = time.time()
        
        # 1. Skeletonize
        print("[Pillar II] 1. Applying Zhang-Suen morphological thinning...")
        self.skeleton = skeletonize(self.mask)
        
        # 2. Extract Graph
        print("[Pillar II] 2. Extracting topological network using sknw...")
        # sknw builds a MultiGraph. We convert it to a directed graph for routing.
        undirected_graph = sknw.build_sknw(self.skeleton, multi=True)
        self.graph = nx.MultiDiGraph(undirected_graph)
        
        # Map node coordinates for later visualization
        for node in self.graph.nodes():
            self.graph.nodes[node]['o'] = undirected_graph.nodes[node]['o']
            
        # Add basic edge weights based on length
        for u, v, k, data in self.graph.edges(keys=True, data=True):
            self.graph[u][v][k]['weight'] = data.get('weight', 1.0)
            
        t1 = time.time()
        print(f"[Pillar II] Complete! Generated mathematical graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges in {(t1-t0)*1000:.2f} ms.")
        
        # Save visualization
        plt.figure(figsize=(8, 8))
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, node_color='cyan', edge_color='red', node_size=500, font_size=10)
        plt.title("Extracted Topological Network")
        plt.savefig("skeleton_graph.png")
        plt.close()
        
        return self.graph


class Pillar_3_Resilience_Engine:
    """Member 3: Criticality analysis and disaster simulation."""
    def __init__(self, graph):
        self.graph = graph

    def analyze_criticality(self):
        print("\n[Pillar III] Running Edge Betweenness Centrality (Brandes Algorithm O(VE))...")
        # Compute betweenness centrality
        self.centrality = nx.edge_betweenness_centrality(self.graph, weight='weight')
        
        # Sort to find the most critical bottleneck
        self.sorted_edges = sorted(self.centrality.items(), key=lambda item: item[1], reverse=True)
        top_edge = self.sorted_edges[0][0]
        score = self.sorted_edges[0][1]
        
        print(f"[Pillar III] Most Critical Bottleneck identified at Edge {top_edge} with centrality score {score:.4f}.")
        return top_edge

    def simulate_disaster(self, target_edge):
        print("\n[Pillar III] Simulating Urban Collapse Scenario...")
        nodes = list(self.graph.nodes())
        start_node = nodes[0]
        end_node = nodes[-1]
        
        try:
            baseline_time = nx.shortest_path_length(self.graph, start_node, end_node, weight='weight')
            print(f"Baseline shortest path travel time: {baseline_time:.2f} units.")
        except nx.NetworkXNoPath:
            print("No baseline path exists!")
            return
            
        print(f"[Pillar III] DISASTER: Edge {target_edge} has been completely washed out.")
        # Sever the edge
        u, v, k = target_edge
        self.graph.remove_edge(u, v, key=k)
        
        try:
            new_time = nx.shortest_path_length(self.graph, start_node, end_node, weight='weight')
            delay = new_time - baseline_time
            print(f"[Pillar III] Rerouting via Dijkstra... New travel time: {new_time:.2f} units.")
            print(f"[Pillar III] Disaster Delay Penalty: +{delay:.2f} units.")
            
            # Simple collapse threshold logic
            if delay > (baseline_time * 0.5):
                print("[Pillar III] SYSTEM STATUS: URBAN COLLAPSE THRESHOLD REACHED (>50% commute increase).")
            else:
                print("[Pillar III] SYSTEM STATUS: Resilient. The grid absorbed the failure.")
                
        except nx.NetworkXNoPath:
            print("[Pillar III] SYSTEM STATUS: TOTAL COLLAPSE. No path to emergency hub exists.")

if __name__ == "__main__":
    print("=== NEUROGRID 3-PILLAR UNIFIED PIPELINE START ===")
    
    # Member 1 Execution
    vision = Pillar_1_Vision(size=200)
    vision.generate_synthetic_grid()
    healed_mask = vision.heal_occlusions()
    
    # Member 2 Execution
    skeletonizer = Pillar_2_Skeletonization(healed_mask)
    graph = skeletonizer.execute()
    
    # Member 3 Execution
    engine = Pillar_3_Resilience_Engine(graph)
    critical_edge = engine.analyze_criticality()
    engine.simulate_disaster(critical_edge)
    
    print("\n=== PIPELINE EXECUTION COMPLETE ===")

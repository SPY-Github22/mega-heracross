import osmnx as ox
import networkx as nx
import folium

def generate_actual_heatmap():
    print("Downloading graph for Folium mapping...")
    G = ox.graph_from_point((12.9279, 77.6271), dist=1500, network_type="drive")
    
    # Calculate degree centrality for coloring
    centrality = nx.degree_centrality(G)
    
    # Create map
    m = folium.Map(location=[12.9279, 77.6271], zoom_start=14, tiles='CartoDB dark_matter')
    
    print("Mapping nodes and edges...")
    # Add edges
    for u, v, data in G.edges(data=True):
        if 'geometry' in data:
            coords = [(lat, lon) for lon, lat in data['geometry'].coords]
        else:
            u_node = G.nodes[u]
            v_node = G.nodes[v]
            coords = [(u_node['y'], u_node['x']), (v_node['y'], v_node['x'])]
            
        folium.PolyLine(coords, color='cyan', weight=1, opacity=0.5).add_to(m)
        
    # Highlight top 10 choke points
    sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    for node_id, cent in sorted_nodes:
        node = G.nodes[node_id]
        folium.CircleMarker(
            location=[node['y'], node['x']],
            radius=cent * 1000, # Scale for visibility
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.7,
            popup=f"Choke Point Node: {node_id}<br>Centrality: {cent:.4f}"
        ).add_to(m)
        
    output_file = "disaster_heatmap.html"
    m.save(output_file)
    print(f"Heatmap successfully saved to {output_file}")

if __name__ == "__main__":
    generate_actual_heatmap()

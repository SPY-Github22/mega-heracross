"""Visualization engine for Part C: Folium heatmaps with collapse overlay."""
import folium
from branca.colormap import LinearColormap
import numpy as np
import os

def build_folium_map(G_full, edge_betweenness, node_criticality,
                     top_n=10, bbox=None, collapse_removed_nodes=None):
    # Centre
    if bbox:
        min_lon, min_lat, max_lon, max_lat = bbox
        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2
    else:
        lats = [G_full.nodes[n]['lat'] for n in G_full.nodes()]
        lons = [G_full.nodes[n]['lon'] for n in G_full.nodes()]
        center_lat = np.mean(lats)
        center_lon = np.mean(lons)

    m = folium.Map(location=[center_lat, center_lon], zoom_start=14,
                   tiles='CartoDB dark_matter', control_scale=True)
    ebc_vals = list(edge_betweenness.values())
    if not ebc_vals:
        return m
    min_ebc, max_ebc = min(ebc_vals), max(ebc_vals)
    colormap = LinearColormap(colors=['blue', 'cyan', 'yellow', 'red'],
                              vmin=min_ebc, vmax=max_ebc,
                              caption='Edge Betweenness Centrality')
    for (u, v), ebc_val in edge_betweenness.items():
        edge_data = G_full.get_edge_data(u, v)
        geometry = edge_data.get('geometry', [])
        if not geometry:
            lat1, lon1 = G_full.nodes[u]['lat'], G_full.nodes[u]['lon']
            lat2, lon2 = G_full.nodes[v]['lat'], G_full.nodes[v]['lon']
            geometry = [(lat1, lon1), (lat2, lon2)]
        folium.PolyLine(locations=[(lat, lon) for lat, lon in geometry],
                        color=colormap(ebc_val), weight=4, opacity=0.8,
                        tooltip=f"EBC: {ebc_val:.4f}").add_to(m)

    sorted_nodes = sorted(node_criticality.items(), key=lambda x: x[1], reverse=True)
    for node, crit in sorted_nodes[:top_n]:
        lat, lon = G_full.nodes[node]['lat'], G_full.nodes[node]['lon']
        radius = 8 + 15 * (crit / max_ebc) if max_ebc > 0 else 8
        folium.CircleMarker(location=[lat, lon], radius=radius,
                            color='red', fill=True, fill_color='red', fill_opacity=0.7,
                            popup=f"<b>Node {node}</b><br>Criticality: {crit:.4f}").add_to(m)

    if collapse_removed_nodes:
        collapse_group = folium.FeatureGroup(name='Collapse Scenario (Ablation)', show=False)
        for node in collapse_removed_nodes:
            lat, lon = G_full.nodes[node]['lat'], G_full.nodes[node]['lon']
            folium.Marker(location=[lat, lon],
                          icon=folium.DivIcon(html='<div style="color:red; font-size:18px; font-weight:bold;">✖</div>'),
                          popup=f"Removed Node {node}").add_to(collapse_group)
        lost_edges = set()
        for u, v in G_full.edges():
            if u in collapse_removed_nodes or v in collapse_removed_nodes:
                lost_edges.add((u, v))
        for u, v in lost_edges:
            edge_data = G_full.get_edge_data(u, v)
            geometry = edge_data.get('geometry', [])
            if not geometry:
                lat1, lon1 = G_full.nodes[u]['lat'], G_full.nodes[u]['lon']
                lat2, lon2 = G_full.nodes[v]['lat'], G_full.nodes[v]['lon']
                geometry = [(lat1, lon1), (lat2, lon2)]
            folium.PolyLine(locations=[(lat, lon) for lat, lon in geometry],
                            color='red', weight=2.5, dash_array='6,6', opacity=0.8,
                            tooltip=f"Lost edge ({u}-{v})").add_to(collapse_group)
        collapse_group.add_to(m)

    folium.LayerControl().add_to(m)
    colormap.add_to(m)
    return m

def generate_static_heatmap(G_full, edge_betweenness, node_criticality,
                            output_path, top_n=10, bbox=None, collapse_removed_nodes=None):
    m = build_folium_map(G_full, edge_betweenness, node_criticality, top_n, bbox, collapse_removed_nodes)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    m.save(output_path)
    print(f"✓ Enhanced heatmap saved to {output_path}")

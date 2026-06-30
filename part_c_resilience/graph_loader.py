"""Graph loading, contract validation, and OSMnx fallback for Part C."""
import json
import logging
import networkx as nx
import osmnx as ox
import os
from shapely.geometry import LineString

from shared.schema import RoadGraph, GraphNode, GraphEdge

logger = logging.getLogger("part_c")

# Set OSMnx timeout
try:
    ox.settings.timeout = 30
    ox.settings.log_console = True
except AttributeError:
    ox.config(timeout=30, log_console=True)

# ── Validation ──────────────────────────────────────────────────
def _validate_type(value, expected_type, field_name):
    if not isinstance(value, expected_type):
        raise ValueError(
            f"Field '{field_name}' must be of type {expected_type.__name__}, "
            f"got {type(value).__name__}."
        )

def _validate_coordinate(lon, lat, node_id, coord_name):
    if not (-180.0 <= lon <= 180.0):
        raise ValueError(f"Node {node_id}: {coord_name} longitude out of range: {lon}")
    if not (-90.0 <= lat <= 90.0):
        raise ValueError(f"Node {node_id}: {coord_name} latitude out of range: {lat}")

def validate_road_graph(data: dict) -> RoadGraph:
    """Thoroughly validate a dict against the RoadGraph schema."""
    for key in ("nodes", "edges", "crs"):
        if key not in data:
            raise ValueError(f"Missing top-level key: '{key}'")

    _validate_type(data["nodes"], list, "nodes")
    _validate_type(data["edges"], list, "edges")
    _validate_type(data["crs"], str, "crs")

    if data["crs"] != "EPSG:4326":
        raise ValueError(f"CRS must be 'EPSG:4326', got '{data['crs']}'")

    node_ids = set()
    nodes = []
    for i, node in enumerate(data["nodes"]):
        if not isinstance(node, dict):
            raise ValueError(f"Node at index {i} is not a dict")
        for field in ("id", "lat", "lon"):
            if field not in node:
                raise ValueError(f"Node {i} missing field '{field}'")
        _validate_type(node["id"], int, f"nodes[{i}].id")
        _validate_type(node["lat"], (int, float), f"nodes[{i}].lat")
        _validate_type(node["lon"], (int, float), f"nodes[{i}].lon")
        _validate_coordinate(node["lon"], node["lat"], node["id"], "Node")
        if node["id"] in node_ids:
            raise ValueError(f"Duplicate node id {node['id']}")
        node_ids.add(node["id"])
        nodes.append(GraphNode(id=node["id"], lat=float(node["lat"]), lon=float(node["lon"])))

    edges = []
    for j, edge in enumerate(data["edges"]):
        if not isinstance(edge, dict):
            raise ValueError(f"Edge at index {j} is not a dict")
        for field in ("source", "target", "weight_m", "geometry"):
            if field not in edge:
                raise ValueError(f"Edge {j} missing field '{field}'")
        _validate_type(edge["source"], int, f"edges[{j}].source")
        _validate_type(edge["target"], int, f"edges[{j}].target")
        _validate_type(edge["weight_m"], (int, float), f"edges[{j}].weight_m")
        _validate_type(edge["geometry"], list, f"edges[{j}].geometry")
        src, tgt = edge["source"], edge["target"]
        if src not in node_ids:
            raise ValueError(f"Edge {j} source {src} does not exist in nodes")
        if tgt not in node_ids:
            raise ValueError(f"Edge {j} target {tgt} does not exist in nodes")

        geometry = []
        for k, point in enumerate(edge["geometry"]):
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError(
                    f"Edge {j} geometry point {k} must be a [lat, lon] pair"
                )
            lat, lon = float(point[0]), float(point[1])
            _validate_coordinate(lon, lat, f"edge {j} geom point {k}", "Geometry")
            geometry.append((lat, lon))
        edges.append(GraphEdge(source=src, target=tgt,
                               weight_m=float(edge["weight_m"]),
                               geometry=geometry))

    return RoadGraph(nodes=nodes, edges=edges, crs=data["crs"])

# ── Graph construction ──────────────────────────────────────────
def load_graph(graph_path: str) -> nx.Graph:
    with open(graph_path, "r") as f:
        raw = json.load(f)
    road_graph = validate_road_graph(raw)
    G = nx.Graph()
    for node in road_graph.nodes:
        G.add_node(node.id, lat=node.lat, lon=node.lon)
    for edge in road_graph.edges:
        G.add_edge(edge.source, edge.target,
                   weight_m=edge.weight_m,
                   geometry=edge.geometry)
    logger.info(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

def fallback_to_osmnx(bbox: tuple, output_path: str = None) -> nx.Graph:
    logger.info("Falling back to OSMnx for road network download...")
    G_osmnx = ox.graph_from_bbox(bbox=bbox, network_type='drive')
    logger.info(f"Downloaded OSMnx graph with {len(G_osmnx.nodes)} nodes, {len(G_osmnx.edges)} edges")
    try:
        G_undirected = ox.convert.to_undirected(G_osmnx)
    except AttributeError:
        G_undirected = ox.utils_graph.get_undirected(G_osmnx)

    nodes = []
    for osm_id, data in G_undirected.nodes(data=True):
        nodes.append(GraphNode(id=osm_id, lat=data['y'], lon=data['x']))

    edges = []
    for u, v, data in G_undirected.edges(data=True):
        length = data.get('length', 0.0)
        if 'geometry' in data and isinstance(data['geometry'], LineString):
            geom = [(lat, lon) for lon, lat in data['geometry'].coords]
        else:
            lat_u, lon_u = G_undirected.nodes[u]['y'], G_undirected.nodes[u]['x']
            lat_v, lon_v = G_undirected.nodes[v]['y'], G_undirected.nodes[v]['x']
            geom = [(lat_u, lon_u), (lat_v, lon_v)]
        edges.append(GraphEdge(source=u, target=v, weight_m=length, geometry=geom))

    road_graph = RoadGraph(nodes=nodes, edges=edges, crs="EPSG:4326")
    road_dict = {
        "nodes": [{"id": n.id, "lat": n.lat, "lon": n.lon} for n in road_graph.nodes],
        "edges": [{"source": e.source, "target": e.target, "weight_m": e.weight_m, "geometry": e.geometry} for e in road_graph.edges],
        "crs": "EPSG:4326"
    }
    validate_road_graph(road_dict)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(road_dict, f, indent=2)
        logger.info(f"Fallback graph saved to {output_path}")

    G = nx.Graph()
    for node in road_graph.nodes:
        G.add_node(node.id, lat=node.lat, lon=node.lon)
    for edge in road_graph.edges:
        G.add_edge(edge.source, edge.target, weight_m=edge.weight_m, geometry=edge.geometry)
    return G

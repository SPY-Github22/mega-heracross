"""Custom Brandes edge betweenness centrality implementation."""
import heapq

def _single_source_dijkstra_paths_and_sigmas(G, s, weight='weight_m'):
    dist = {s: 0.0}
    sigma = {s: 1.0}
    pred = {s: []}
    visited = set()
    pq = [(0.0, s)]
    while pq:
        d_u, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        for v in G[u]:
            w = G[u][v].get(weight, 1.0)
            d_v = d_u + w
            if v not in dist or d_v < dist[v] - 1e-12:
                dist[v] = d_v
                sigma[v] = sigma[u]
                pred[v] = [u]
                heapq.heappush(pq, (d_v, v))
            elif abs(d_v - dist[v]) < 1e-12:
                sigma[v] += sigma[u]
                pred[v].append(u)
    return dist, sigma, pred

def brandes_edge_betweenness(G, weight='weight_m', normalized=True):
    n = G.number_of_nodes()
    edge_betweenness = {}
    for u, v in G.edges():
        if u < v:
            edge_betweenness[(u, v)] = 0.0
        else:
            edge_betweenness[(v, u)] = 0.0
    for s in G.nodes():
        dist, sigma, pred = _single_source_dijkstra_paths_and_sigmas(G, s, weight)
        stack = sorted([v for v in dist if v != s], key=lambda x: dist[x], reverse=True)
        delta = {v: 0.0 for v in G.nodes()}
        for v in stack:
            for u in pred[v]:
                contrib = (sigma[u] / sigma[v]) * (1.0 + delta[v])
                edge_key = (u, v) if u < v else (v, u)
                edge_betweenness[edge_key] += contrib
                delta[u] += contrib
    if normalized and n > 1:
        scale = 1.0 / (n * (n - 1))
        for edge in edge_betweenness:
            edge_betweenness[edge] *= scale
    return edge_betweenness

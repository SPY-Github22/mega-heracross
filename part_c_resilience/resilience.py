"""Core resilience analysis: centrality, efficiency, ablation, Monte Carlo, cascading."""
import logging
import random
import time
import networkx as nx
import numpy as np
from scipy import stats
from joblib import Parallel, delayed
from tqdm import tqdm
from math import radians, sin, cos, sqrt, atan2

from shared.config import COLLAPSE_THRESHOLD as DEFAULT_COLLAPSE_THRESHOLD
from brandes import brandes_edge_betweenness

logger = logging.getLogger("part_c")
USE_CUSTOM_BRANDES = True
LARGE_GRAPH_NODE_THRESHOLD = 10000
SAMPLE_PAIRS = 20000

# ── Edge Betweenness ───────────────────────────────────────────
def compute_edge_betweenness(G: nx.Graph, weight='weight_m') -> dict:
    if USE_CUSTOM_BRANDES:
        logger.info("Using custom Brandes algorithm...")
        ebc = brandes_edge_betweenness(G, weight, normalized=True)
    else:
        ebc = nx.edge_betweenness_centrality(G, weight=weight, normalized=True)
    logger.info(f"Edge betweenness computed for {len(ebc)} edges.")
    return ebc

# ── Node Criticality ───────────────────────────────────────────
def compute_node_criticality(G: nx.Graph, edge_betweenness: dict) -> dict:
    """Sum of incident EBC, normalised to [0,1]."""
    raw_sum = {n: 0.0 for n in G.nodes()}
    for (u, v), eb in edge_betweenness.items():
        raw_sum[u] += eb
        raw_sum[v] += eb
    max_sum = max(raw_sum.values()) if raw_sum else 1.0
    if max_sum == 0:
        return {n: 0.0 for n in G.nodes()}
    node_crit = {n: s / max_sum for n, s in raw_sum.items()}
    logger.info("Node criticality (sum of incident EBC, norm 0-1) calculated.")
    return node_crit

# ── Global Efficiency ──────────────────────────────────────────
def global_efficiency(G, weight='weight_m', sample=False, sample_pairs=None):
    n = G.number_of_nodes()
    if n < 2:
        return 0.0
    if sample or n > LARGE_GRAPH_NODE_THRESHOLD:
        if sample_pairs is None:
            logger.info(f"Large graph ({n} nodes) detected; using approximate efficiency with {SAMPLE_PAIRS} sample pairs.")
            sp = sample_node_pairs(G, SAMPLE_PAIRS)
        else:
            sp = sample_pairs
        eff, sem = approximate_global_efficiency(G, sp)
        logger.info(f"Approximate global efficiency: {eff:.6e} ± {sem:.6e}")
        return eff
    else:
        total_inv_dist = 0.0
        for node in G.nodes():
            lengths = nx.single_source_dijkstra_path_length(G, node, weight=weight)
            for v, d in lengths.items():
                if node != v:
                    total_inv_dist += 1.0 / d
        E = total_inv_dist / (n * (n - 1))
        logger.info(f"Exact global efficiency: {E:.6e}")
        return E

# ── Delay & Normalised RI ──────────────────────────────────────
def travel_delay_penalty(current_efficiency, baseline_efficiency):
    if baseline_efficiency <= 0 or current_efficiency <= 0:
        return float('inf')
    return (baseline_efficiency / current_efficiency) - 1.0

def normalized_resilience_index(current_efficiency, baseline_efficiency):
    if baseline_efficiency <= 0:
        return 0.0
    return current_efficiency / baseline_efficiency

def compute_baseline(G):
    E0 = global_efficiency(G)
    logger.info(f"Baseline Resilience Index (E): {E0:.6f}")
    return E0

# ── Sampling ───────────────────────────────────────────────────
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def get_nodes_within_radius(G, center_node, radius_m):
    lat_c = G.nodes[center_node]['lat']
    lon_c = G.nodes[center_node]['lon']
    nearby = []
    for node in G.nodes():
        if node == center_node:
            continue
        if haversine_distance(lat_c, lon_c, G.nodes[node]['lat'], G.nodes[node]['lon']) <= radius_m:
            nearby.append(node)
    return nearby

def sample_node_pairs(G, num_pairs, rng=None):
    nodes = list(G.nodes())
    N = len(nodes)
    if N < 2:
        return []
    if rng is None:
        rng = np.random.default_rng(42)
    total_possible = N * (N - 1) // 2
    if num_pairs >= total_possible:
        pairs = [(nodes[i], nodes[j]) for i in range(N) for j in range(i+1, N)]
    else:
        idx_pairs = set()
        while len(idx_pairs) < num_pairs:
            i = rng.integers(0, N)
            j = rng.integers(0, N)
            if i != j:
                a, b = (i, j) if i < j else (j, i)
                idx_pairs.add((a, b))
        pairs = [(nodes[a], nodes[b]) for a, b in idx_pairs]
    sampled_distances = []
    logger.info(f"Computing baseline distances for {len(pairs)} sample pairs...")
    for u, v in tqdm(pairs, desc="Sample distances"):
        try:
            d = nx.shortest_path_length(G, u, v, weight='weight_m')
        except nx.NetworkXNoPath:
            d = float('inf')
        sampled_distances.append((u, v, d))
    return sampled_distances

def approximate_global_efficiency(G, sampled_pairs):
    inv_ds = [1.0/d for _, _, d in sampled_pairs if d > 0 and d < float('inf')]
    if not inv_ds:
        return 0.0, 0.0
    mean_inv = np.mean(inv_ds)
    sem = np.std(inv_ds, ddof=1) / np.sqrt(len(inv_ds)) if len(inv_ds) > 1 else 0.0
    return mean_inv, sem

def precompute_baseline_distances(G, use_sample=None):
    n = G.number_of_nodes()
    if use_sample is None:
        use_sample = (n > LARGE_GRAPH_NODE_THRESHOLD)
    if use_sample:
        logger.info(f"Large graph: sampling {SAMPLE_PAIRS} pairs for baseline distances.")
        sampled = sample_node_pairs(G, SAMPLE_PAIRS)
        base_dist = {(u, v): d for u, v, d in sampled}
        return base_dist, sampled
    else:
        dist = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight_m'))
        flat_dist = {}
        for u, d_dict in dist.items():
            for v, d in d_dict.items():
                if u < v:
                    flat_dist[(u, v)] = d
        return flat_dist, None

# ── Ablation ───────────────────────────────────────────────────
def run_node_ablation(G_original, node_criticality, baseline_efficiency,
                      use_sample=False, collapse_threshold=DEFAULT_COLLAPSE_THRESHOLD):
    sorted_nodes = sorted(node_criticality.items(), key=lambda x: x[1], reverse=True)
    G = G_original.copy()
    trajectory = []
    collapse_result = {"collapse_removals": None, "collapse_efficiency": None,
                       "collapse_penalty": None, "collapse_normalized_ri": None,
                       "trajectory": trajectory}
    for i, (node, crit) in enumerate(sorted_nodes):
        G.remove_node(node)
        removals = i + 1
        if G.number_of_nodes() < 2:
            current_e = 0.0
        else:
            current_e = global_efficiency(G, sample=use_sample)
        penalty = travel_delay_penalty(current_e, baseline_efficiency)
        norm_ri = normalized_resilience_index(current_e, baseline_efficiency)
        trajectory.append((removals, current_e, penalty, norm_ri))
        if penalty > collapse_threshold and collapse_result["collapse_removals"] is None:
            collapse_result["collapse_removals"] = removals
            collapse_result["collapse_efficiency"] = current_e
            collapse_result["collapse_penalty"] = penalty
            collapse_result["collapse_normalized_ri"] = norm_ri
            break
    if collapse_result["collapse_removals"] is None:
        collapse_result["collapse_removals"] = len(sorted_nodes)
        if trajectory:
            last = trajectory[-1]
            collapse_result["collapse_efficiency"] = last[1]
            collapse_result["collapse_penalty"] = last[2]
            collapse_result["collapse_normalized_ri"] = last[3]
    return collapse_result

# ── Monte Carlo Chaos ──────────────────────────────────────────
def _survival_rate_and_penalty_exact(G_original, damaged_G, removed_nodes, total_pairs):
    """Exact computation for small graphs (requires original all-pairs distances)."""
    # Not used in latest version; kept for legacy
    pass

def _survival_rate_and_penalty_sampled(G_original, damaged_G, removed_nodes, pairs_list):
    connected_pairs = 0
    penalty_sum = 0.0
    penalty_count = 0
    for u, v in pairs_list:
        if u in removed_nodes or v in removed_nodes:
            continue
        try:
            d_base = nx.shortest_path_length(G_original, u, v, weight='weight_m')
        except nx.NetworkXNoPath:
            continue
        try:
            d_damaged = nx.shortest_path_length(damaged_G, u, v, weight='weight_m')
        except nx.NetworkXNoPath:
            continue
        connected_pairs += 1
        penalty = (d_damaged / d_base) - 1.0
        penalty_sum += penalty
        penalty_count += 1
    active_pairs = [ (u,v) for (u,v) in pairs_list if u not in removed_nodes and v not in removed_nodes ]
    total_active = len(active_pairs)
    survival_rate = connected_pairs / total_active if total_active > 0 else 0.0
    avg_penalty = penalty_sum / penalty_count if penalty_count > 0 else 0.0
    return survival_rate, avg_penalty

def monte_carlo_chaos_simulation(G, baseline_dist=None,
                                 num_scenarios=100, k=5,
                                 baseline_efficiency=None,
                                 n_jobs=1,
                                 pairs_list=None):
    nodes = list(G.nodes())
    rng = random.Random(42)
    removed_sets = [rng.sample(nodes, k) for _ in range(num_scenarios)]

    if pairs_list is not None:
        def _scenario(removed):
            temp_G = G.copy()
            temp_G.remove_nodes_from(removed)
            return _survival_rate_and_penalty_sampled(G, temp_G, removed, pairs_list)
    else:
        # exact mode (using baseline_dist)
        if baseline_dist is None:
            raise ValueError("For exact Monte Carlo, baseline_dist dict is required.")
        total_pairs = G.number_of_nodes() * (G.number_of_nodes() - 1) // 2
        def _scenario_exact(removed):
            temp_G = G.copy()
            temp_G.remove_nodes_from(removed)
            # full all-pairs on damaged graph
            dist_damaged = dict(nx.all_pairs_dijkstra_path_length(temp_G, weight='weight_m'))
            connected = 0
            penalty_sum = 0.0
            penalty_count = 0
            remaining = [n for n in G.nodes() if n not in removed]
            for i, u in enumerate(remaining):
                for v in remaining[i+1:]:
                    if (u, v) in baseline_dist or (v, u) in baseline_dist:
                        d_base = baseline_dist.get((u, v), baseline_dist.get((v, u)))
                    else:
                        continue
                    if v in dist_damaged.get(u, {}):
                        d_dam = dist_damaged[u][v]
                        connected += 1
                        penalty_sum += (d_dam / d_base) - 1.0
                        penalty_count += 1
            survival = connected / total_pairs if total_pairs > 0 else 0.0
            pen = penalty_sum / penalty_count if penalty_count > 0 else 0.0
            return survival, pen
        _scenario = _scenario_exact

    logger.info(f"Running {num_scenarios} Monte Carlo scenarios (k={k}) with n_jobs={n_jobs}...")
    t0 = time.time()
    results = Parallel(n_jobs=n_jobs)(delayed(_scenario)(rem) for rem in removed_sets)
    t = time.time() - t0

    survival_rates = [r[0] for r in results]
    penalties = [r[1] for r in results]

    def ci95(data):
        arr = np.array(data)
        mean = np.mean(arr)
        sem = stats.sem(arr)
        interval = 1.96 * sem
        return mean, interval

    mean_sr, ci_sr = ci95(survival_rates)
    mean_pen, ci_pen = ci95(penalties)

    logger.info(f"Monte Carlo results: survival {mean_sr*100:.1f}% ± {ci_sr*100:.1f}%, delay penalty {mean_pen*100:.1f}%")
    return {
        "survival_rates": survival_rates,
        "penalties": penalties,
        "mean_survival": mean_sr,
        "ci_survival": ci_sr,
        "mean_penalty": mean_pen,
        "ci_penalty": ci_pen,
    }

# ── Cascading Failures ─────────────────────────────────────────
def monte_carlo_cascading_simulation(G_original, baseline_efficiency, node_criticality,
                                     num_scenarios=20, initial_k=1, radius_m=200,
                                     correlation_prob=0.8,
                                     cascade_threshold=DEFAULT_COLLAPSE_THRESHOLD,
                                     max_cascade_steps=10):
    rng = random.Random(42)
    nodes_list = list(G_original.nodes())
    results = []
    logger.info(f"Running {num_scenarios} cascading failure scenarios...")
    t0 = time.time()
    for _ in range(num_scenarios):
        seeds = rng.sample(nodes_list, initial_k)
        failed = set(seeds)
        for seed in seeds:
            nearby = get_nodes_within_radius(G_original, seed, radius_m)
            for n in nearby:
                if rng.random() < correlation_prob:
                    failed.add(n)
        G_damaged = G_original.copy()
        G_damaged.remove_nodes_from(failed)
        cascade_steps = 0
        while cascade_steps < max_cascade_steps and G_damaged.number_of_nodes() >= 2:
            curr_eff = global_efficiency(G_damaged)
            penalty = travel_delay_penalty(curr_eff, baseline_efficiency)
            if penalty > cascade_threshold:
                break
            ebc = compute_edge_betweenness(G_damaged)
            crit = compute_node_criticality(G_damaged, ebc)
            if not crit:
                break
            top_node = max(crit, key=crit.get)
            G_damaged.remove_node(top_node)
            cascade_steps += 1
        total_removed = len(failed) + cascade_steps
        acceleration = cascade_steps / len(failed) if len(failed) > 0 else float('inf')
        results.append({
            'initial_failures': len(failed),
            'cascade_length': cascade_steps,
            'total_removed': total_removed,
            'acceleration_factor': acceleration
        })
    t = time.time() - t0
    logger.info(f"Cascading simulation completed in {t:.1f}s")
    agg = {
        'mean_cascade_length': np.mean([r['cascade_length'] for r in results]),
        'median_cascade_length': np.median([r['cascade_length'] for r in results]),
        'mean_total_removed': np.mean([r['total_removed'] for r in results]),
        'mean_acceleration': np.mean([r['acceleration_factor'] for r in results if r['acceleration_factor'] != float('inf')]),
        'scenarios': results
    }
    return agg

# ── Judge Report ───────────────────────────────────────────────
def generate_judge_report(G_orig, node_crit, ablation_results, mc_results,
                          baseline_efficiency, cascade_results=None):
    nodes = G_orig.number_of_nodes()
    edges = G_orig.number_of_edges()
    top_node = max(node_crit.items(), key=lambda x: x[1]) if node_crit else (None, 0.0)
    top_id, top_val = top_node
    collapse_removals = ablation_results.get('collapse_removals', 0)
    collapse_penalty = ablation_results.get('collapse_penalty', 0.0)
    mean_sr = mc_results.get('mean_survival', 0.0)
    ci_sr = mc_results.get('ci_survival', 0.0)
    mean_pen = mc_results.get('mean_penalty', 0.0)

    summary = (
        f"Koramangala network: {nodes} nodes, {edges} edges. "
        f"Top choke point: node {top_id} (centrality {top_val:.4f}). "
        f"Collapse at {collapse_removals}-node removal. "
        f"Monte Carlo survival rate: {mean_sr*100:.1f}% ± {ci_sr*100:.1f}%. "
        f"Mean delay penalty: {mean_pen*100:.1f}%"
    )
    if cascade_results:
        summary += (
            f" Cascading failure: median cascade length {cascade_results['median_cascade_length']:.1f}, "
            f"mean acceleration {cascade_results['mean_acceleration']:.2f}x."
        )
    metrics = {
        "nodes": nodes,
        "edges": edges,
        "top_choke_node": top_id,
        "top_choke_centrality": round(top_val, 6),
        "collapse_removals": collapse_removals,
        "collapse_penalty": round(collapse_penalty, 6),
        "monte_carlo_survival_rate": round(mean_sr, 6),
        "monte_carlo_survival_ci": round(ci_sr, 6),
        "monte_carlo_delay_penalty": round(mean_pen, 6),
        "baseline_efficiency": round(baseline_efficiency, 6),
        "summary": summary
    }
    if cascade_results:
        metrics.update({
            "cascade_mean_length": cascade_results['mean_cascade_length'],
            "cascade_mean_acceleration": cascade_results['mean_acceleration'],
        })
    return summary, metrics

# ── Sensitivity ────────────────────────────────────────────────
def sensitivity_k_sweep(G, baseline_eff, k_values, n_scenarios=25):
    logger.info(f"Running sensitivity sweep for k = {k_values} ({n_scenarios} scenarios each)...")
    t0 = time.time()
    mean_sr, ci_sr, mean_pen, ci_pen = [], [], [], []
    baseline_dist, _ = precompute_baseline_distances(G)
    for k in k_values:
        mc = monte_carlo_chaos_simulation(G, baseline_dist,
                                          num_scenarios=n_scenarios, k=k,
                                          baseline_efficiency=baseline_eff,
                                          n_jobs=1)
        mean_sr.append(mc['mean_survival'])
        ci_sr.append(mc['ci_survival'])
        mean_pen.append(mc['mean_penalty'])
        ci_pen.append(mc['ci_penalty'])
    t = time.time() - t0
    logger.info(f"Sensitivity sweep completed in {t:.1f}s")
    return {
        'k_values': k_values,
        'survival_mean': mean_sr,
        'survival_ci': ci_sr,
        'penalty_mean': mean_pen,
        'penalty_ci': ci_pen,
    }

def sensitivity_threshold_sweep(G, node_criticality, baseline_eff, thresholds):
    sorted_nodes = sorted(node_criticality.items(), key=lambda x: x[1], reverse=True)
    removal_counts = []
    for thresh in thresholds:
        G_temp = G.copy()
        for i, (node, _) in enumerate(sorted_nodes):
            G_temp.remove_node(node)
            if G_temp.number_of_nodes() < 2:
                removal_counts.append(i+1)
                break
            curr_eff = global_efficiency(G_temp)
            penalty = travel_delay_penalty(curr_eff, baseline_eff)
            if penalty > thresh:
                removal_counts.append(i+1)
                break
        else:
            removal_counts.append(len(sorted_nodes))
    return {'thresholds': thresholds, 'removal_counts': removal_counts}

def run_resilience_analysis(G):
    ebc = compute_edge_betweenness(G)
    node_crit = compute_node_criticality(G, ebc)
    return ebc, node_crit

def print_top_choke_points(node_crit, ebc, top_n=10):
    sorted_nodes = sorted(node_crit.items(), key=lambda x: x[1], reverse=True)
    sorted_edges = sorted(ebc.items(), key=lambda x: x[1], reverse=True)
    print(f"\n── Top {top_n} Critical Nodes ──")
    for rank, (n, c) in enumerate(sorted_nodes[:top_n], 1):
        print(f"  {rank:2d}. Node {n:6d} | Criticality: {c:.6f}")
    print(f"\n── Top {top_n} Critical Edges ──")
    for rank, ((u,v), c) in enumerate(sorted_edges[:top_n], 1):
        print(f"  {rank:2d}. Edge ({u:6d}, {v:6d}) | Betweenness: {c:.6f}")

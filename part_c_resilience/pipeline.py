"""Full analysis pipeline for Part C, used by both headless and dashboard."""
import os, json, logging
from resilience import (
    run_resilience_analysis, compute_baseline, run_node_ablation,
    precompute_baseline_distances, monte_carlo_chaos_simulation,
    monte_carlo_cascading_simulation, generate_judge_report
)
from viz import generate_static_heatmap

logger = logging.getLogger("part_c")

def run_resilience_pipeline(G, config):
    output_dir = config['output_dir']
    bbox = tuple(config['bbox'])
    heatmap_path = config['heatmap_path']
    collapse_threshold = config['collapse_threshold']

    os.makedirs(output_dir, exist_ok=True)

    # Phase 4
    ebc, node_crit = run_resilience_analysis(G)

    # Phase 5
    baseline_eff = compute_baseline(G)

    # Phase 6
    ablation_results = run_node_ablation(G, node_crit, baseline_eff,
                                         collapse_threshold=collapse_threshold)

    # Phase 7: precompute baseline distances & Monte Carlo
    baseline_dist, _ = precompute_baseline_distances(G)
    mc_results = monte_carlo_chaos_simulation(
        G, baseline_dist,
        num_scenarios=config['monte_carlo_scenarios'],
        k=config['monte_carlo_k'],
        baseline_efficiency=baseline_eff,
        n_jobs=1
    )

    # Phase 20: Cascading
    cascade_results = monte_carlo_cascading_simulation(
        G, baseline_eff, node_crit,
        num_scenarios=config['cascading_scenarios'],
        initial_k=1,
        radius_m=config['cascading_radius_m'],
        correlation_prob=config['cascading_correlation_prob'],
        cascade_threshold=collapse_threshold
    )
    G.graph['cascade_results'] = cascade_results

    # Phase 8: judge report
    summary, metrics = generate_judge_report(G, node_crit, ablation_results,
                                             mc_results, baseline_eff, cascade_results)
    print("\n" + "=" * 60)
    print("         JUDGE-READY SCORE REPORT")
    print("=" * 60)
    print(summary)
    print("=" * 60)

    eval_path = os.path.join(output_dir, "evaluation.json")
    with open(eval_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Evaluation metrics saved to {eval_path}")

    # Phase 9/10: heatmap
    sorted_crit = sorted(node_crit.items(), key=lambda x: x[1], reverse=True)
    removed_at_collapse = [n for n, _ in sorted_crit[:ablation_results['collapse_removals']]]
    generate_static_heatmap(
        G, ebc, node_crit,
        output_path=heatmap_path,
        top_n=10,
        bbox=bbox,
        collapse_removed_nodes=removed_at_collapse
    )

    metrics['heatmap_path'] = heatmap_path
    return metrics

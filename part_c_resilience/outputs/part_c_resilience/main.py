"""
Part C – Resilience & Visualization Engine
Main entry point for headless execution.
"""
import sys, os, json

# Inject project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils import load_config, logger
from graph_loader import load_graph, fallback_to_osmnx
from pipeline import run_resilience_pipeline

def main():
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    config = load_config(config_path)
    logger.info(f"City: {config['city_name']}, BBOX: {config['bbox']}")

    # Load graph
    if config['graph_source'] == 'part_b' and os.path.exists(config['graph_path']):
        G = load_graph(config['graph_path'])
    else:
        G = fallback_to_osmnx(tuple(config['bbox']), output_path=config['graph_path'])

    # Run pipelines
    metrics = run_resilience_pipeline(G, config)

    print("\nAll outputs saved to", config['output_dir'])
    print("Heatmap:", config['heatmap_path'])
    print("Evaluation:", os.path.join(config['output_dir'], "evaluation.json"))

if __name__ == "__main__":
    main()

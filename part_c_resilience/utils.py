"""Configuration loader and logging setup for Part C."""
import logging
import sys
import yaml
import os

def setup_logging(level=logging.INFO):
    logger = logging.getLogger("part_c")
    if not logger.handlers:
        logger.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%H:%M:%S"
        )
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = setup_logging()

def load_config(config_path=None):
    """Load configuration from YAML, with fallback to shared/config.py defaults."""
    from shared.config import (
        TARGET_CRS, COLLAPSE_THRESHOLD, TEST_TILE_BBOX, GRAPH_SOURCE,
        GRAPH_PATH, HEATMAP_PATH, ROAD_MASK_PATH, META_PATH
    )
    # Defaults
    config = {
        'city_name': 'Koramangala, Bengaluru',
        'bbox': list(TEST_TILE_BBOX),
        'collapse_threshold': COLLAPSE_THRESHOLD,
        'monte_carlo_scenarios': 100,
        'monte_carlo_k': 5,
        'cascading_scenarios': 20,
        'cascading_radius_m': 200,
        'cascading_correlation_prob': 0.8,
        'random_seed': 42,
        'graph_source': GRAPH_SOURCE,
        'output_dir': 'part_c_resilience/outputs',
        'graph_path': GRAPH_PATH,
        'heatmap_path': HEATMAP_PATH,
        'road_mask_path': ROAD_MASK_PATH,
        'meta_path': META_PATH,
        'target_crs': TARGET_CRS
    }

    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f) or {}
        for key in config:
            if key in user_config:
                config[key] = user_config[key]
        if 'collapse_threshold' in user_config and user_config['collapse_threshold'] != COLLAPSE_THRESHOLD:
            logger.warning(f"Config override: collapse_threshold {user_config['collapse_threshold']} differs from contract {COLLAPSE_THRESHOLD}. Using config value.")
        if 'bbox' in user_config and tuple(user_config['bbox']) != TEST_TILE_BBOX:
            logger.info(f"Custom bbox: {user_config['bbox']}")
    return config

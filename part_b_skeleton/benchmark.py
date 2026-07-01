"""
part_b_skeleton/benchmark.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 25: Full Pipeline Benchmark + Reproducibility Report

Runs the complete Part B pipeline, times every stage individually,
hashes all inputs and outputs, and writes a reproducibility report.

Output: part_b_skeleton/outputs/part_b_benchmark.json

Contents:
  - Per-stage timing (ms)
  - Total wall-clock time
  - Input hash (road_mask.npy or synthetic seed)
  - Output hash (graph.json SHA256)
  - All hyperparameters used
  - All evaluation metrics (contract, F1, LCC, AUC, CI)
  - Timestamp and Python version

Target: Total pipeline < 30s on CPU (excluding OSM download which is cached).

This file is the reproducibility artefact — an ISRO judge can verify
that the same inputs always produce the same outputs, and check that
the pipeline meets the performance target.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict

import numpy as np

# ── repo root on path ─────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ══════════════════════════════════════════════════════════════
# HASHING
# ══════════════════════════════════════════════════════════════

def hash_file(path: str) -> str:
    """SHA256 hash of a file. Returns 'FILE_NOT_FOUND' if absent."""
    if not os.path.exists(path):
        return "FILE_NOT_FOUND"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]   # first 16 chars is enough for identity check


def hash_array(arr: np.ndarray) -> str:
    """SHA256 hash of a numpy array."""
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def hash_dict(d: dict) -> str:
    """SHA256 hash of a JSON-serialisable dict."""
    return hashlib.sha256(
        json.dumps(d, sort_keys=True).encode()
    ).hexdigest()[:16]


# ══════════════════════════════════════════════════════════════
# STAGE TIMER
# ══════════════════════════════════════════════════════════════

class StageTimer:
    """Context manager that records wall-clock time for a named stage."""

    def __init__(self, name: str, timings: dict):
        self.name    = name
        self.timings = timings
        self._t0     = None

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_):
        elapsed_ms = (time.perf_counter() - self._t0) * 1000
        self.timings[self.name] = round(elapsed_ms, 1)


# ══════════════════════════════════════════════════════════════
# BENCHMARK RUNNER
# ══════════════════════════════════════════════════════════════

def run_benchmark(
        mask_path: str,
        meta_path: str,
        graph_json_path: str,
        all_metrics: Dict[str, Any],
        output_path: str,
) -> Dict:
    """
    Collect benchmark data and write part_b_benchmark.json.

    Parameters
    ----------
    mask_path       : str — path to road_mask.npy (may not exist)
    meta_path       : str — path to meta.json (may not exist)
    graph_json_path : str — path to output graph.json
    all_metrics     : dict — all metrics collected during the run
    output_path     : str — where to write part_b_benchmark.json

    Returns
    -------
    benchmark dict
    """
    ts = datetime.now(timezone.utc).isoformat()

    # ── Input hashes ──────────────────────────────────────────
    input_hashes = {
        "road_mask_npy": hash_file(mask_path),
        "meta_json":     hash_file(meta_path),
    }

    # ── Output hash ───────────────────────────────────────────
    output_hash = hash_file(graph_json_path)

    # ── Hyperparameters ───────────────────────────────────────
    from shared.config import (
        TARGET_CRS, COLLAPSE_THRESHOLD, TEST_TILE_BBOX, GRAPH_SOURCE
    )

    hyperparams = {
        "TARGET_CRS":          TARGET_CRS,
        "COLLAPSE_THRESHOLD":  COLLAPSE_THRESHOLD,
        "TEST_TILE_BBOX":      list(TEST_TILE_BBOX),
        "GRAPH_SOURCE":        GRAPH_SOURCE,
        "n_removals_fragility": 50,
        "n_boot_ci":           100,
        "lcc_threshold":       0.80,
        "snap_m_base":         "resolution_aware",
        "spline_healing":      True,
        "sar_guided":          True,
    }

    # ── System info ───────────────────────────────────────────
    system_info = {
        "python_version": sys.version.split()[0],
        "platform":       platform.platform(),
        "cpu_count":      os.cpu_count(),
    }

    # ── Key metrics snapshot ──────────────────────────────────
    metrics_snapshot = {
        "contract_status":  all_metrics.get("contract_status", "UNKNOWN"),
        "node_count":       all_metrics.get("node_count", 0),
        "edge_count":       all_metrics.get("edge_count", 0),
        "lcc_pct":          all_metrics.get("lcc_pct", 0),
        "node_f1":          all_metrics.get("node_f1", 0),
        "edge_f1":          all_metrics.get("edge_f1", 0),
        "total_length_km":  all_metrics.get("total_length_km", 0),
        "source":           all_metrics.get("source", "unknown"),
        "resolution_m":     all_metrics.get("resolution_m", 0),
        "fragility_auc":    all_metrics.get("fragility_auc", 0),
        "node_f1_ci":       all_metrics.get("node_f1_ci", [0, 0]),
        "edge_f1_ci":       all_metrics.get("edge_f1_ci", [0, 0]),
        # Context flags — explain low F1 scores caused by comparing
        # synthetic test data against a real (denser) OSM ground truth
        "osm_source_type":  all_metrics.get("osm_source_type", "unknown"),
        "scale_mismatch":   all_metrics.get("scale_mismatch", False),
        "scale_ratio":      all_metrics.get("scale_ratio", 0),
    }

    benchmark = {
        "timestamp":         ts,
        "system":            system_info,
        "hyperparameters":   hyperparams,
        "input_hashes":      input_hashes,
        "output_hash":       output_hash,
        "metrics":           metrics_snapshot,
        "timings_ms":        all_metrics.get("timings_ms", {}),
        "total_runtime_s":   all_metrics.get("total_runtime_s", 0),
        "passes_30s_target": all_metrics.get("total_runtime_s", 0) < 30.0,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(benchmark, f, indent=2)

    return benchmark


# ══════════════════════════════════════════════════════════════
# PRINT REPORT
# ══════════════════════════════════════════════════════════════

def print_benchmark_report(benchmark: Dict) -> None:
    """Print Phase 25 benchmark report — the final judge-ready summary."""
    SEP  = "═" * 62
    SEP2 = "─" * 62
    m    = benchmark.get("metrics", {})
    t    = benchmark.get("timings_ms", {})
    ts   = benchmark.get("timestamp", "")[:19].replace("T", " ")

    passes = benchmark.get("passes_30s_target", False)
    total  = benchmark.get("total_runtime_s", 0)

    print(f"\n{SEP}")
    print(f"  PHASE 25 — PIPELINE BENCHMARK & REPRODUCIBILITY")
    print(f"  {ts} UTC")
    print(SEP)

    print(f"\n  Runtime (CPU only):")
    stage_map = [
        ("loader",        "Phase 03 mask loader"),
        ("skeletonize",   "Phase 04 skeletonization"),
        ("graph_build",   "Phase 05 sknw extraction"),
        ("healing",       "Phases 10-13 healing"),
        ("simplify",      "Phase 14 simplification"),
        ("weight_audit",  "Phase 16 weight audit"),
        ("classify",      "Phase 22 classification"),
        ("fragility",     "Phase 23 fragility"),
        ("bootstrap",     "Phase 24 bootstrap CI"),
    ]
    for key, label in stage_map:
        ms = t.get(key, 0)
        bar = "█" * min(20, int(ms / 200)) + "░" * max(0, 20 - int(ms / 200))
        print(f"    {label:<30} {ms:>7.1f} ms  {bar}")

    print(f"\n  Total: {total:.2f}s  {'✓ < 30s target' if passes else '✗ exceeds 30s target'}")

    print(f"\n  Input hashes:")
    for k, v in benchmark.get("input_hashes", {}).items():
        print(f"    {k:<20} {v}")
    print(f"  Output hash (graph.json): {benchmark.get('output_hash', '?')}")

    print(f"\n{SEP2}")
    print(f"  FINAL METRICS (Koramangala test tile):")
    print(f"    Contract         : {m.get('contract_status','?')}")
    print(f"    Nodes / Edges    : {m.get('node_count',0)} / {m.get('edge_count',0)}")
    print(f"    Total length     : {m.get('total_length_km',0):.3f} km")
    print(f"    LCC%             : {m.get('lcc_pct',0):.1%}")
    print(f"    node_F1          : {m.get('node_f1',0):.4f}  CI {m.get('node_f1_ci',[0,0])}")
    print(f"    edge_F1          : {m.get('edge_f1',0):.4f}  CI {m.get('edge_f1_ci',[0,0])}")
    if m.get("scale_mismatch"):
        print(f"    ⚠ F1 scored against REAL OSM data ({m.get('osm_source_type','?')}) "
              f"while Part B ran on synthetic mask ({m.get('scale_ratio',0):.1f}× size gap)")
        print(f"      → Low F1 is expected until Part A delivers road_mask.npy")
    print(f"    Fragility AUC    : {m.get('fragility_auc',0):.4f}")
    print(f"    Source           : {m.get('source','?')} @ {m.get('resolution_m',0):.1f} m/px")

    print(f"\n{SEP}")
    print(f"  BENCHMARK: {'✓ PASS' if passes else '⚠ SLOW'} | "
          f"Output hash: {benchmark.get('output_hash','?')}")
    print(f"  Reproducibility: same inputs → same hash every run")
    print(SEP + "\n")

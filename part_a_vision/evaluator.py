# part_a_vision/evaluator.py
#
# Part A Evaluation Module - Mega-Heracross BAH 2026
#
# This module owns all measurement for Part A outputs.
# It is imported by shared/eval.py and called by run_part_a.py (Phase 24).
#
# What it measures:
#   Pixel-level:    IoU, Precision, Recall, F1
#   Topology:       Skeleton IoU, Edge F1
#   Connectivity:   Connected component count, Largest CC fraction
#   Contract:       dtype, shape, value range, meta field presence
#
# Design principle:
#   Every function that computes a metric also explains WHY that metric
#   matters for the BAH use case (urban road resilience). These explanations
#   appear in comments and in the judge report text.

import os
import sys
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
from scipy.ndimage import label, sobel

# Skimage for skeletonization - available from requirements.txt
try:
    from skimage.morphology import thin as _skimage_thin
    _SKIMAGE_AVAILABLE = True
except ImportError:
    _SKIMAGE_AVAILABLE = False

# ── Path setup ────────────────────────────────────────────────────────────────
_here      = Path(__file__).resolve().parent
_repo_root = _here.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from shared.config import (
    TARGET_CRS, TEST_TILE_BBOX,
    ROAD_MASK_PATH, META_PATH,
)
from part_a_vision.part_a_config import (
    OSMNX_GT_MASK_PATH,
    SMALL_CC_FILTER_PX,
    EVAL_REPORT_PATH,
)


# ------------------------------------------------------------------------------
# SECTION 1 - CONTRACT VALIDATION
# ------------------------------------------------------------------------------

def validate_contract(
    mask_path: str = ROAD_MASK_PATH,
    meta_path: str = META_PATH,
) -> Dict[str, Any]:
    """
    Load and validate Part A outputs against the shared contract.

    This is a HARD validator: any contract violation raises ValueError immediately.
    It is intentionally stricter than output_writer.py's write-time checks because
    it also validates the combination of mask + meta (e.g. CRS consistency).

    Returns:
        dict with 'mask' (np.ndarray) and 'meta' (dict) if both pass.

    Raises:
        FileNotFoundError: file does not exist
        ValueError: contract violation with exact description and fix hint
    """
    errors = []

    # ── road_mask.npy ────────────────────────────────────────────────────────
    mask_path = str(mask_path)
    if not os.path.exists(mask_path):
        raise FileNotFoundError(
            f"[evaluator] road_mask.npy not found: '{mask_path}'\n"
            f"  Fix: Run Part A pipeline first."
        )

    mask = np.load(mask_path)

    if mask.dtype != np.uint8:
        errors.append(
            f"  road_mask.npy dtype: expected uint8, got {mask.dtype}\n"
            f"    Fix: mask = (prob_map > 0.5).astype(np.uint8)"
        )

    if mask.ndim != 2:
        errors.append(
            f"  road_mask.npy ndim: expected 2 (H,W), got {mask.ndim} {mask.shape}\n"
            f"    Fix: squeeze extra dimensions before saving"
        )

    if mask.ndim == 2:
        unique_vals = set(np.unique(mask).tolist())
        if not unique_vals.issubset({0, 1}):
            errors.append(
                f"  road_mask.npy values: expected only {{0,1}}, got {unique_vals}\n"
                f"    Fix: ensure model output is thresholded at 0.5 and cast to uint8"
            )

    if mask.ndim == 2 and (mask.shape[0] == 0 or mask.shape[1] == 0):
        errors.append(f"  road_mask.npy has zero-length dimension: {mask.shape}")

    # ── meta.json ─────────────────────────────────────────────────────────────
    meta_path = str(meta_path)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"[evaluator] meta.json not found: '{meta_path}'\n"
            f"  Fix: Run Part A pipeline first."
        )

    with open(meta_path, 'r') as f:
        meta = json.load(f)

    required_fields = {'crs', 'bbox', 'resolution_m', 'source'}
    missing = required_fields - set(meta.keys())
    if missing:
        errors.append(
            f"  meta.json missing fields: {missing}\n"
            f"    Present: {set(meta.keys())}"
        )

    if 'crs' in meta and meta['crs'] != TARGET_CRS:
        errors.append(
            f"  meta.json crs: expected '{TARGET_CRS}', got '{meta['crs']}'\n"
            f"    Fix: always reproject inputs to EPSG:4326 before processing"
        )

    if 'bbox' in meta:
        bbox = meta['bbox']
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            errors.append(f"  meta.json bbox: expected 4-element list, got {bbox!r}")
        else:
            min_lon, min_lat, max_lon, max_lat = bbox
            if not (min_lon < max_lon):
                errors.append(f"  meta.json bbox: min_lon ({min_lon}) >= max_lon ({max_lon})")
            if not (min_lat < max_lat):
                errors.append(f"  meta.json bbox: min_lat ({min_lat}) >= max_lat ({max_lat})")

    if 'resolution_m' in meta:
        res = meta['resolution_m']
        if not isinstance(res, (int, float)) or res <= 0:
            errors.append(f"  meta.json resolution_m: must be positive float, got {res!r}")

    if 'source' in meta and (not isinstance(meta['source'], str) or not meta['source'].strip()):
        errors.append(f"  meta.json source: must be non-empty string, got {meta['source']!r}")

    # ── Fail if any errors found ───────────────────────────────────────────────
    if errors:
        raise ValueError(
            "[evaluator] Contract validation FAILED:\n" +
            "\n".join(errors)
        )

    print("[OK] Contract validation PASSED")
    return {'mask': mask, 'meta': meta}


# ------------------------------------------------------------------------------
# SECTION 2 - GROUND TRUTH LOADER
# ------------------------------------------------------------------------------

def load_gt_mask(
    meta:    Dict[str, Any],
    gt_path: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Load the ground truth mask for evaluation.

    Priority order:
      1. Explicit gt_path argument (highest trust)
      2. OSMNX_GT_MASK_PATH (generated in Phase 2 - always available)
      3. Generate from OSMnx on the fly (slowest, but self-healing)
      4. Return None with warning (metrics will be skipped)

    In Phase 2 (synthetic mode):
        GT = OSMnx-rasterized road mask = same as road_mask.npy
        Expected IoU = 1.0 (trivially perfect - no model involved yet)
        This is CORRECT behaviour: the GT mask IS the prediction in Phase 2.

    From Phase 7 onwards:
        GT = OSMnx GT (for Koramangala evaluation)
        GT = DeepGlobe labels (for DeepGlobe validation split)
        IoU will drop to a realistic value (~0.5-0.85 depending on phase)
    """
    # Option 1: explicit path
    if gt_path is not None:
        gt_path = str(gt_path)
        if os.path.exists(gt_path):
            gt = np.load(gt_path)
            _assert_valid_binary_mask(gt, "provided gt_path")
            print(f"  GT loaded from: {gt_path}")
            return gt
        else:
            warnings.warn(f"[evaluator] gt_path not found: '{gt_path}'. Falling back.")

    # Option 2: cached OSMnx GT (Phase 2 creates this)
    osmnx_gt = str(OSMNX_GT_MASK_PATH)
    if os.path.exists(osmnx_gt):
        gt = np.load(osmnx_gt)
        _assert_valid_binary_mask(gt, OSMNX_GT_MASK_PATH)
        print(f"  GT loaded from OSMnx cache: {osmnx_gt}")
        source = meta.get('source', '')
        if source == 'synthetic':
            print("  Note: source='synthetic' → GT = OSMnx mask = road_mask.npy")
            print("        IoU = 1.0 is expected until Phase 7 model replaces the prediction.")
        return gt

    # Option 3: generate on the fly
    print("  OSMnx GT cache not found. Generating from OSMnx...")
    try:
        from part_a_vision.synthetic_tile import SyntheticTileGenerator
        gen = SyntheticTileGenerator()
        gt  = gen._get_gt_mask()
        print(f"  GT generated from OSMnx (not cached).")
        return gt
    except Exception as e:
        warnings.warn(f"[evaluator] Could not generate GT: {e}")

    # Option 4: no GT available
    warnings.warn(
        "[evaluator] No ground truth available. "
        "Pixel metrics will be skipped. "
        "Run Phase 2 (synthetic_tile.py --mode demo) to create osmnx_gt_mask.npy."
    )
    return None


def _assert_valid_binary_mask(mask: np.ndarray, name: str) -> None:
    """Raise if mask is not a valid binary uint8 2D array."""
    if mask.dtype != np.uint8:
        raise ValueError(f"[evaluator] {name} dtype is {mask.dtype}, expected uint8")
    if mask.ndim != 2:
        raise ValueError(f"[evaluator] {name} ndim is {mask.ndim}, expected 2")
    unique = set(np.unique(mask).tolist())
    if not unique.issubset({0, 1}):
        raise ValueError(f"[evaluator] {name} has values {unique}, expected {{0,1}}")


# ------------------------------------------------------------------------------
# SECTION 3 - PIXEL-LEVEL METRICS
# ------------------------------------------------------------------------------

def compute_metrics(
    pred: np.ndarray,
    gt:   np.ndarray,
) -> Dict[str, Any]:
    """
    Compute pixel-level segmentation metrics between pred and gt.

    Both must be binary uint8 arrays with identical shapes.

    Metrics:
        IoU (Jaccard):  TP / (TP + FP + FN)
            Measures overlap between predicted and true road pixels.
            The primary metric for road extraction benchmarks (DeepGlobe uses this).
            Range: 0 (no overlap) to 1 (perfect overlap).

        Precision:      TP / (TP + FP)
            "Of all pixels I called road, what fraction actually are roads?"
            High precision = few false road alarms. Low precision = predicting
            roads where there are none (false positives).

        Recall:         TP / (TP + FN)
            "Of all actual road pixels, what fraction did I find?"
            High recall = found most roads. For disaster response, recall matters
            more than precision: missing a critical road is more dangerous than
            a false alarm.

        F1:             2*TP / (2*TP + FP + FN)
            Harmonic mean of Precision and Recall.
            Better than arithmetic mean for imbalanced classes.
            For road pixels (~15% of image), F1 is more informative than accuracy.

    Edge cases:
        gt all-zero (no roads): all metrics = NaN. This tile has no roads to detect.
            A model that predicts all-background is not useful here.
        Both all-zero (no roads, no predictions): all metrics = NaN.
            Perfect agreement on emptiness is trivially true; avoid inflating scores.
        pred all-zero (no predictions made): IoU=0, P=0, R=0, F1=0.
            The model is completely wrong if there are actual roads.

    Returns:
        dict with keys: iou, precision, recall, f1, tp, fp, fn,
                        road_px_pred, road_px_gt, total_px, road_fraction_gt
    """
    _assert_valid_binary_mask(pred, "pred")
    _assert_valid_binary_mask(gt,   "gt")

    if pred.shape != gt.shape:
        raise ValueError(
            f"[compute_metrics] Shape mismatch: pred={pred.shape}, gt={gt.shape}\n"
            f"  Fix: resize pred to match gt before evaluation."
        )

    gt_sum = int(gt.sum())

    # Edge case: gt has no road pixels → metrics are undefined
    if gt_sum == 0:
        warnings.warn(
            "[compute_metrics] gt_mask has no road pixels (all-zero). "
            "All metrics are NaN. This tile has no roads to detect."
        )
        nan = float('nan')
        return {
            'iou': nan, 'precision': nan, 'recall': nan, 'f1': nan,
            'tp': 0, 'fp': int(pred.sum()), 'fn': 0,
            'road_px_pred': int(pred.sum()), 'road_px_gt': 0,
            'total_px': pred.size, 'road_fraction_gt': 0.0,
        }

    # Cast to int32 for arithmetic (avoid uint8 overflow)
    p = pred.astype(np.int32)
    g = gt.astype(np.int32)

    TP = int((p * g).sum())
    FP = int((p * (1 - g)).sum())
    FN = int(((1 - p) * g).sum())

    # Numerically stable: add epsilon to all denominators
    eps = 1e-8

    iou       = TP / (TP + FP + FN + eps)
    precision = TP / (TP + FP + eps)
    recall    = TP / (TP + FN + eps)
    f1        = (2 * TP) / (2 * TP + FP + FN + eps)

    return {
        'iou':              float(iou),
        'precision':        float(precision),
        'recall':           float(recall),
        'f1':               float(f1),
        'tp':               TP,
        'fp':               FP,
        'fn':               FN,
        'road_px_pred':     int(pred.sum()),
        'road_px_gt':       gt_sum,
        'total_px':         int(pred.size),
        'road_fraction_gt': float(gt_sum / pred.size),
    }


# ------------------------------------------------------------------------------
# SECTION 4 - TOPOLOGY METRICS
# ------------------------------------------------------------------------------

def count_connected_components(
    mask:        np.ndarray,
    min_area_px: int = SMALL_CC_FILTER_PX,
) -> Dict[str, Any]:
    """
    Count connected road components in a binary mask.

    Why connectivity matters for BAH:
        A road network's utility depends on it being one connected graph.
        100 isolated road segments is NOT a useful road network for routing.
        For disaster resilience analysis (Part C), a fragmented road network
        has artificially high "criticality" for every node - misleading results.

    We use 4-connectivity (up/down/left/right only, not diagonal).
    Why not 8-connectivity?
        Roads are linear horizontal/vertical features.
        8-connectivity can merge parallel roads through a single diagonal pixel,
        making distinct roads appear as one component.

    We filter out components smaller than min_area_px pixels.
    Why?
        Small isolated components are model noise - single misclassified pixels.
        Including them in the CC count makes the metric noisy and misleading.

    Returns dict with:
        count:            number of significant connected components
        largest_fraction: fraction of road pixels in the largest component
                          (closer to 1.0 = more connected network = better)
        sizes:            list of significant component sizes (descending)
        filtered_count:   number of components removed as noise
    """
    _assert_valid_binary_mask(mask, "mask")

    # scipy.ndimage.label with default structure = 4-connectivity for 2D
    labeled, n_total = label(mask)

    if n_total == 0:
        return {
            'count': 0,
            'largest_fraction': 0.0,
            'sizes': [],
            'filtered_count': 0,
        }

    # Component sizes: np.bincount counts pixels per label; skip label 0 (background)
    sizes_all = np.bincount(labeled.ravel())[1:]

    # Filter: only keep components >= min_area_px
    significant = sizes_all[sizes_all >= min_area_px]
    n_significant = int(len(significant))
    n_filtered    = n_total - n_significant

    total_road_px = int(mask.sum())

    if n_significant == 0 or total_road_px == 0:
        return {
            'count': 0,
            'largest_fraction': 0.0,
            'sizes': [],
            'filtered_count': n_filtered,
        }

    largest_size     = int(significant.max())
    largest_fraction = float(largest_size / total_road_px)
    sizes_sorted     = sorted(significant.tolist(), reverse=True)

    return {
        'count':            n_significant,
        'largest_fraction': largest_fraction,
        'sizes':            sizes_sorted,
        'filtered_count':   n_filtered,
    }


def compute_skeleton_iou(
    pred: np.ndarray,
    gt:   np.ndarray,
) -> float:
    """
    Compute Skeleton IoU: IoU between the skeletonized prediction and GT.

    Why Skeleton IoU?
        Pixel IoU measures AREA overlap. A thick road prediction (3px wide)
        on a thin GT (1px wide) gets penalized even though the road is found.
        Skeleton IoU measures TOPOLOGICAL fidelity: does the predicted road
        NETWORK have the same structure (connectivity, junctions, continuity)
        as the GT network?

        For Part B (graph extraction via Zhang-Suen skeletonization), the
        skeleton quality DIRECTLY determines graph quality. Skeleton IoU
        measures exactly what Part B needs.

    Method:
        skimage.morphology.thin() performs iterative thinning (similar to
        Zhang-Suen) to reduce a binary mask to a 1-pixel-wide skeleton.
        We then compute pixel IoU on the two skeletons.

    Expected range: typically 0.5–0.9, always lower than pixel IoU.
    If Skeleton IoU is much lower than pixel IoU, the model finds road AREA
    but misses road CONTINUITY - common failure mode.

    Returns:
        float in [0,1], or NaN if skimage is unavailable or GT is empty.
    """
    if not _SKIMAGE_AVAILABLE:
        warnings.warn(
            "[compute_skeleton_iou] skimage not available. "
            "Install with: pip install scikit-image"
        )
        return float('nan')

    if gt.sum() == 0:
        return float('nan')

    pred_skel = _skimage_thin(pred.astype(bool)).astype(np.uint8)
    gt_skel   = _skimage_thin(gt.astype(bool)).astype(np.uint8)

    result = compute_metrics(pred_skel, gt_skel)
    return result['iou']


def compute_edge_f1(
    pred: np.ndarray,
    gt:   np.ndarray,
    threshold: float = 0.3,
) -> float:
    """
    Compute Edge F1: F1 score comparing road boundary detection.

    Why Edge F1?
        Pixel IoU is agnostic to WHERE within a thick road prediction the
        boundary falls. Edge F1 specifically measures how accurately the
        model predicts the EDGES (outlines) of roads.

        Sharp road edges matter because:
        - Part B's skeletonization is more accurate with sharp edges
        - Road width estimation requires precise edge detection
        - ISRO judges doing visual assessment notice blurry edges

    Method:
        1. Apply Sobel operator (X and Y gradients) to both pred and gt
        2. Compute edge magnitude: |G| = sqrt(Gx^2 + Gy^2)
        3. Threshold to binary edge maps at `threshold * max_magnitude`
        4. Compute F1 between the two binary edge maps

    Sobel operator kernels:
        Gx = [-1, 0, 1; -2, 0, 2; -1, 0, 1]  (horizontal edges)
        Gy = [-1, -2, -1; 0, 0, 0; 1, 2, 1]   (vertical edges)

    threshold = 0.3 means: a pixel is an "edge" if its gradient magnitude
    is at least 30% of the maximum gradient in the image.

    Returns:
        float in [0,1], or NaN if GT edges are empty.
    """
    def _sobel_edges(mask_arr: np.ndarray) -> np.ndarray:
        """Apply Sobel filter and return thresholded binary edge map."""
        m = mask_arr.astype(np.float32)
        gx  = sobel(m, axis=0)   # gradient in row direction
        gy  = sobel(m, axis=1)   # gradient in column direction
        mag = np.hypot(gx, gy)   # edge magnitude: sqrt(gx^2 + gy^2)
        max_mag = mag.max()
        if max_mag < 1e-8:
            return np.zeros_like(mask_arr, dtype=np.uint8)
        return (mag > threshold * max_mag).astype(np.uint8)

    pred_edges = _sobel_edges(pred)
    gt_edges   = _sobel_edges(gt)

    if gt_edges.sum() == 0:
        return float('nan')

    result = compute_metrics(pred_edges, gt_edges)
    return result['f1']


# ------------------------------------------------------------------------------
# SECTION 5 - ORCHESTRATOR
# ------------------------------------------------------------------------------

def build_eval_result(
    mask_path: str = ROAD_MASK_PATH,
    meta_path: str = META_PATH,
    gt_path:   Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the complete Part A evaluation pipeline.

    Steps:
        1. Validate contract (raises on failure)
        2. Load ground truth mask
        3. Compute pixel metrics (IoU, P, R, F1)
        4. Compute connected component analysis
        5. Compute Skeleton IoU
        6. Compute Edge F1
        7. Package everything into a result dict

    Returns:
        Comprehensive evaluation dict. Pass this to print_judge_report()
        or save_eval_report() for output.
    """
    print(f"\n[Part A Evaluator] Running evaluation...")
    print(f"  Prediction: {mask_path}")
    print(f"  Meta:       {meta_path}")

    # ── Step 1: Contract validation ───────────────────────────────────────────
    contract = validate_contract(mask_path, meta_path)
    pred     = contract['mask']
    meta     = contract['meta']

    # ── Step 2: Load ground truth ─────────────────────────────────────────────
    gt = load_gt_mask(meta, gt_path)

    # ── Step 3: Resize pred to match GT if needed ─────────────────────────────
    if gt is not None and pred.shape != gt.shape:
        import cv2
        warnings.warn(
            f"[evaluator] pred.shape {pred.shape} != gt.shape {gt.shape}. "
            f"Resizing pred to match GT using nearest-neighbour interpolation."
        )
        pred_resized = cv2.resize(
            pred, (gt.shape[1], gt.shape[0]),
            interpolation=cv2.INTER_NEAREST
        ).astype(np.uint8)
    else:
        pred_resized = pred

    # ── Step 4: Pixel metrics ─────────────────────────────────────────────────
    if gt is not None:
        pixel_metrics = compute_metrics(pred_resized, gt)
    else:
        nan = float('nan')
        pixel_metrics = {
            'iou': nan, 'precision': nan, 'recall': nan, 'f1': nan,
            'tp': 0, 'fp': 0, 'fn': 0,
            'road_px_pred': int(pred.sum()), 'road_px_gt': 0,
            'total_px': pred.size, 'road_fraction_gt': nan,
        }

    # ── Step 5: Connected component analysis (on PREDICTION) ─────────────────
    # We analyse the predicted mask's connectivity because this is what
    # Part B receives. GT connectivity is a separate reference.
    cc_result = count_connected_components(pred)
    if gt is not None:
        cc_gt = count_connected_components(gt)
    else:
        cc_gt = None

    # ── Step 6: Topology metrics ──────────────────────────────────────────────
    if gt is not None and gt.sum() > 0 and pred.sum() > 0:
        skel_iou = compute_skeleton_iou(pred_resized, gt)
        edge_f1  = compute_edge_f1(pred_resized, gt)
    else:
        skel_iou = float('nan')
        edge_f1  = float('nan')

    # ── Step 7: Package result ────────────────────────────────────────────────
    result = {
        'timestamp':       datetime.now().isoformat(timespec='seconds'),
        'mask_path':       str(mask_path),
        'meta_path':       str(meta_path),
        'gt_path':         str(gt_path) if gt_path else str(OSMNX_GT_MASK_PATH),
        'gt_available':    gt is not None,
        'contract_passed': True,   # we got here, so it passed
        # Meta fields
        'crs':             meta.get('crs', '?'),
        'bbox':            meta.get('bbox', []),
        'resolution_m':    meta.get('resolution_m', 0.0),
        'source':          meta.get('source', '?'),
        # Pixel metrics
        **pixel_metrics,
        # Topology
        'skeleton_iou':    skel_iou,
        'edge_f1':         edge_f1,
        # Connectivity (prediction)
        'cc_count':        cc_result['count'],
        'cc_largest_frac': cc_result['largest_fraction'],
        'cc_sizes':        cc_result['sizes'],
        'cc_noise_count':  cc_result['filtered_count'],
        # GT connectivity (for reference)
        'gt_cc_count':     cc_gt['count'] if cc_gt else None,
        'gt_cc_largest':   cc_gt['largest_fraction'] if cc_gt else None,
    }

    return result


# ------------------------------------------------------------------------------
# SECTION 6 - REPORTING
# ------------------------------------------------------------------------------

def _fmt(val: Any, decimals: int = 3, width: int = 8) -> str:
    """Format a metric value for the report. Handles NaN cleanly."""
    if isinstance(val, float) and (val != val):   # NaN check
        return f"{'N/A':>{width}}"
    if isinstance(val, float):
        return f"{val:.{decimals}f}".rjust(width)
    return str(val).rjust(width)


def print_judge_report(result: Dict[str, Any]) -> None:
    """
    Print the judge-ready score report to stdout.

    Format follows the spec from the project document, with additions:
        - Skeleton IoU (topology)
        - Edge F1 (boundary sharpness)
        - Largest CC fraction (network connectivity health)
        - Uncertainty fraction (Phase 19 will populate this)

    This output is what the ISRO judge sees when Part A runs at the finale.
    Every field has an explanation available - know what each one means.
    """
    sep  = "-" * 60
    dash = "─" * 60

    bbox  = result.get('bbox', [])
    bbox_str = (
        f"({bbox[0]:.4f}-{bbox[2]:.4f}, {bbox[1]:.4f}-{bbox[3]:.4f})"
        if len(bbox) == 4 else "N/A"
    )

    road_pred = result.get('road_px_pred', 0)
    total_px  = result.get('total_px', 0)
    road_gt   = result.get('road_px_gt', 0)

    lines = [
        f"\n{sep}",
        f"  MEGA-HERACROSS Part A - Koramangala Vision Report",
        f"{sep}",
        f"  Tile:          {bbox_str}",
        f"  Source:        {result.get('source','?')} "
            f"| Resolution: {result.get('resolution_m',0):.1f}m/pixel",
        f"  Timestamp:     {result.get('timestamp','?')}",
        f"{dash}",
        f"  PIXEL METRICS",
        f"    IoU:           {_fmt(result.get('iou', float('nan')), 3, 6)}",
        f"    Precision:     {_fmt(result.get('precision', float('nan')), 3, 6)}",
        f"    Recall:        {_fmt(result.get('recall', float('nan')), 3, 6)}",
        f"    F1:            {_fmt(result.get('f1', float('nan')), 3, 6)}",
        f"{dash}",
        f"  TOPOLOGY METRICS",
        f"    Skeleton IoU:  {_fmt(result.get('skeleton_iou', float('nan')), 3, 6)}",
        f"    Edge F1:       {_fmt(result.get('edge_f1', float('nan')), 3, 6)}",
        f"{dash}",
        f"  CONNECTIVITY (prediction)",
        f"    Connected CCs: {result.get('cc_count', 0):>6d}",
        f"    Largest CC:    {_fmt(result.get('cc_largest_frac', float('nan')), 1, 5)}% of road pixels",
        f"    Noise CCs:     {result.get('cc_noise_count', 0):>6d}  (< {SMALL_CC_FILTER_PX}px, filtered)",
    ]

    # GT connectivity reference (if available)
    if result.get('gt_cc_count') is not None:
        lines.append(
            f"  CONNECTIVITY (GT reference)"
        )
        lines.append(
            f"    GT CCs:        {result['gt_cc_count']:>6d}"
        )
        if result.get('gt_cc_largest') is not None:
            lines.append(
                f"    GT Largest CC: {result['gt_cc_largest']*100:.1f}% of GT road pixels"
            )

    lines += [
        f"{dash}",
        f"  COVERAGE",
        f"    Pred road px:  {road_pred:>8,} / {total_px:>8,} "
            f"({road_pred/max(total_px,1)*100:.1f}%)",
        f"    GT road px:    {road_gt:>8,} / {total_px:>8,} "
            f"({road_gt/max(total_px,1)*100:.1f}%)",
    ]

    # Uncertainty (Phase 19 adds this; show N/A as a placeholder)
    unc_frac = result.get('uncertainty_fraction', None)
    if unc_frac is not None:
        lines.append(
            f"    Uncertain px:  {_fmt(unc_frac, 1, 5)}% of road pixels"
        )

    contract_str = "[OK] PASSED" if result.get('contract_passed') else "✗ FAILED"
    gt_note = "(OSMnx reference)" if result.get('source') == 'synthetic' else ""

    lines += [
        f"{dash}",
        f"  Contract:      {contract_str}",
        f"  GT source:     OSMnx-rasterized Koramangala {gt_note}",
        f"{sep}\n",
    ]

    report_str = "\n".join(lines)
    print(report_str)


def save_eval_report(
    result:      Dict[str, Any],
    report_path: str = EVAL_REPORT_PATH,
) -> None:
    """
    Save the eval report to a text file with timestamp.
    Appends to the file (one run per session) rather than overwriting.
    This lets you track metric progression across phases.
    """
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Run: {result.get('timestamp', 'unknown')}\n")
        f.write(f"{'='*60}\n")
        # Write key metrics as simple key=value lines for easy parsing
        for key in ['iou', 'precision', 'recall', 'f1', 'skeleton_iou',
                    'edge_f1', 'cc_count', 'cc_largest_frac',
                    'road_px_pred', 'road_px_gt', 'source']:
            val = result.get(key, 'N/A')
            if isinstance(val, float) and (val != val):
                val = 'NaN'
            f.write(f"  {key}: {val}\n")
        f.write(f"  contract_passed: {result.get('contract_passed', False)}\n")

    print(f"  Eval report appended: {report_path}")


def run_part_a_evaluation(
    mask_path:   str           = ROAD_MASK_PATH,
    meta_path:   str           = META_PATH,
    gt_path:     Optional[str] = None,
    report_path: Optional[str] = EVAL_REPORT_PATH,
) -> Dict[str, Any]:
    """
    Top-level entry point. Run this after every Part A pipeline execution.

    Orchestrates: validate → load GT → compute metrics → print report → save.

    Returns:
        eval_result dict (same as build_eval_result output).
    """
    result = build_eval_result(mask_path, meta_path, gt_path)
    print_judge_report(result)
    if report_path:
        save_eval_report(result, report_path)
    return result
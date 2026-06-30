# part_a_vision/part_a_config.py
# Part A internal configuration.
# Imports locked constants from shared/config.py and adds Part A-specific settings.
# You are allowed to tune these - they don't affect the contract.

import os
import sys

# ── Path resolution ────────────────────────────────────────────────────────────
# This ensures `from shared.config import ...` works regardless of where
# you run the script from, as long as you're somewhere inside the project.
_here = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_here)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# ── Import locked constants ────────────────────────────────────────────────────
from shared.config import (
    TARGET_CRS,
    COLLAPSE_THRESHOLD,
    TEST_TILE_BBOX,
    GRAPH_SOURCE,
    ROAD_MASK_PATH,
    META_PATH,
    GRAPH_PATH,
    HEATMAP_PATH,
)

# ── Model Architecture ─────────────────────────────────────────────────────────
# Primary: SegFormer B3 (45M params, hierarchical transformer)
# Fallback: DeepLabV3+ with ResNet-50 backbone
# To switch: change MODEL_BACKBONE here - the model loader checks this string.
MODEL_BACKBONE = "segformer_b3"       # "deeplabv3plus_resnet50" as fallback

# ── Input Configuration ────────────────────────────────────────────────────────
TILE_SIZE         = 512               # pixels - SegFormer requires multiples of 32
N_OPTICAL_BANDS   = 4                 # LISS-IV: Green, Red, NIR, SWIR
N_SAR_BANDS       = 2                 # Sentinel-1: VV, VH polarization
N_INDEX_BANDS     = 0                 # Phase 22 adds 3 (NDVI, NDWI, RPI)
N_INPUT_CHANNELS  = N_OPTICAL_BANDS + N_SAR_BANDS + N_INDEX_BANDS  # = 6 now, 9 in Phase 22
N_CLASSES         = 2                 # binary: road (1) vs background (0)

# ── Training Hyperparameters ───────────────────────────────────────────────────
BATCH_SIZE        = 8
LEARNING_RATE     = 1e-4              # default LR (DeepLabV3+ / single-LR training)
ENCODER_LR        = 6e-5             # SegFormer encoder LR (slower - pretrained)
DECODER_LR        = 6e-4             # SegFormer decoder LR (faster - trained from scratch)
WEIGHT_DECAY      = 1e-4
N_EPOCHS          = 50
EARLY_STOP_PATIENCE = 10             # stop if val IoU doesn't improve for 10 epochs

# ── Class Imbalance ────────────────────────────────────────────────────────────
# Roads are ~15-20% of pixels in urban imagery.
# Without this weight, BCE will converge to predicting all-background.
# weight_road = 1 / road_fraction = 1 / 0.17 ≈ 6.0
ROAD_PIXEL_WEIGHT = 6.0

# ── Loss Function Weights ──────────────────────────────────────────────────────
# These four weights sum to 1.0.
# Phases 8, 9, 10 introduce each term progressively.
# Final formula: L = 0.4*Dice + 0.3*BCE + 0.2*Boundary + 0.1*Connectivity
LOSS_WEIGHTS = {
    "dice":         0.4,
    "bce":          0.3,
    "boundary":     0.2,
    "connectivity": 0.1,
}

# ── Synthetic Generation ───────────────────────────────────────────────────────
SYNTHETIC_SEED              = 42
SYNTHETIC_N_TRAIN_TILES     = 200
SYNTHETIC_N_VAL_TILES       = 40
SYNTHETIC_OCCLUSION_MAX     = 0.70    # max fraction of tile that can be occluded
SYNTHETIC_RESOLUTION_M      = 5.8    # match LISS-IV resolution

# ── Post-processing ────────────────────────────────────────────────────────────
# Phase 18 uses these.
# Conservative values - aggressive post-processing can merge distinct roads.
MORPHOLOGY_CLOSE_RADIUS     = 3       # pixels - fills gaps smaller than this
MIN_ROAD_COMPONENT_PX       = 50     # remove isolated components below this area

# ── Evaluation ────────────────────────────────────────────────────────────────
SMALL_CC_FILTER_PX          = 10     # connected components below this are noise
UNCERTAINTY_THRESHOLD       = 0.20   # pixels with MC Dropout std > this are uncertain

# ── Inference ────────────────────────────────────────────────────────────────
USE_TTA                     = False   # Test-Time Augmentation (8x slower, +0.02 IoU)
TTA_N_AUGMENTATIONS         = 8
UNCERTAINTY_MODE            = "tta_variance"   # "mc_dropout" | "tta_variance" | "none"
MC_DROPOUT_SAMPLES          = 30
ONNX_OPSET_VERSION          = 14

# ── Paths (Part A internal) ────────────────────────────────────────────────────
OSMNX_CACHE_PATH            = "part_a_vision/data/koramangala/koramangala_osmnx.graphml"
OSMNX_GT_MASK_PATH          = "part_a_vision/data/koramangala/osmnx_gt_mask.npy"
BEST_CHECKPOINT_PATH        = "part_a_vision/models/best_checkpoint.pth"
ONNX_MODEL_PATH             = "part_a_vision/models/segformer_b3.onnx"
TRAIN_LOG_PATH              = "part_a_vision/logs/train_log.csv"
EVAL_REPORT_PATH            = "part_a_vision/logs/eval_report.txt"
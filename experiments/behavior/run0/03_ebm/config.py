"""
Configuration for behavior run0 / 03_ebm.

Same task split and data handling as 01_fit.
EBM-specific hyperparameters live here.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
METADATA     = "stimuli/imgs/metadata.jsonl"
CACHE_PATH   = "stimuli/imgs/clip_rn50_features_v2.pt"
RESOURCES    = Path(__file__).parent.parent / "resources"
BEHAVIOR_CSV = RESOURCES / "dlbt-behavior.csv"
RESULTS_DIR  = Path(__file__).parent / "results"

# ---------------------------------------------------------------------------
# Data handling
# ---------------------------------------------------------------------------
SEED        = 42
SEEDS       = [42]

USE_TRIAL_KINDS    = ("main", "probe")
MIN_CATCH_PERF     = 1.0
MAIN_PERF_QUANTILE = 0.95
EVAL_CELL_FRAC     = 0.10          # must match 01_fit to get the same split

# ---------------------------------------------------------------------------
# EBM model
# ---------------------------------------------------------------------------
N_MC_SAMPLES  = 2000       # fixed uniform simplex samples (more → smoother estimate)
COMPRESS_DIM  = 128        # CLIP 1024 → compress_dim
HIDDEN_DIM    = 256        # compress_dim + K → hidden_dim → 1
MC_SEED       = 0          # seed for the fixed MC sample set

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
N_EPOCHS         = 1000
PATIENCE         = 100
LR               = 1e-3
INNER_BATCH_SIZE = 32      # images per inner mini-batch (memory knob)
GRAD_CLIP        = 1.0
ENT_WEIGHT       = 0.1     # entropy-reg strength; keeps ESS/N ≥ ~0.2

RUN_TAG = "ebm"

# ---------------------------------------------------------------------------
# Task split (identical to 01_fit / 02_data_efficiency)
# ---------------------------------------------------------------------------
TRAIN_TASKS = [
    "right", "transparent", "glossy", "large",
    "left", "opaque", "matte", "small",
    "right_and_transparent", "left_and_transparent",
    "right_and_glossy",      "left_and_glossy",
    "transparent_and_glossy",
    "large_and_transparent", "large_and_glossy",
    "right_and_transparent_and_glossy",
    "left_and_transparent_and_glossy",
    "large_and_transparent_and_glossy",
]
VAL_TASKS = [
    "right_and_large",
    "left_and_large",
    "right_and_large_and_glossy",
    "right_and_large_and_transparent",
]

BEH_ID_TO_TASK = {
    "transparent":              "transparent",
    "opaque":                   "opaque",
    "glossy":                   "glossy",
    "matte":                    "matte",
    "large":                    "large",
    "small":                    "small",
    "right":                    "right",
    "left":                     "left",
    "transparent,right":        "right_and_transparent",
    "transparent,left":         "left_and_transparent",
    "glossy,right":             "right_and_glossy",
    "glossy,left":              "left_and_glossy",
    "transparent,glossy":       "transparent_and_glossy",
    "transparent,large":        "large_and_transparent",
    "glossy,large":             "large_and_glossy",
    "transparent,glossy,right": "right_and_transparent_and_glossy",
    "transparent,glossy,left":  "left_and_transparent_and_glossy",
    "transparent,glossy,large": "large_and_transparent_and_glossy",
    "large,right":              "right_and_large",
    "large,left":               "left_and_large",
    "glossy,large,right":       "right_and_large_and_glossy",
    "transparent,large,right":  "right_and_large_and_transparent",
}

# ---------------------------------------------------------------------------
# Plot colours
# ---------------------------------------------------------------------------
C_TRAIN = "#E76F51"
C_EVAL  = "#F4A261"
C_STIM  = "#457B9D"
C_TASK  = "#9B5DE5"
C_JOINT = "#43AA8B"
C_EBM   = "#2D6A4F"

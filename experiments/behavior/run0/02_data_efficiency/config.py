"""
Configuration for behavior run0 / 02_data_efficiency.

Identical task split and model settings as 01_fit.  Training budgets are
varied from 10 to the full 90% training set.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
METADATA    = "stimuli/imgs/metadata.jsonl"
CACHE_PATH  = "stimuli/imgs/clip_rn50_features_v2.pt"
RESOURCES   = Path(__file__).parent.parent / "resources"
BEHAVIOR_CSV = RESOURCES / "dlbt-behavior.csv"
RESULTS_DIR  = Path(__file__).parent / "results"

# ---------------------------------------------------------------------------
# Data handling
# ---------------------------------------------------------------------------
SEED        = 42
N_SEEDS     = 1
SEEDS       = [42]

USE_TRIAL_KINDS = ("main", "probe")

MIN_CATCH_PERF     = 1.0
MAIN_PERF_QUANTILE = 0.95

# 10% of (main × TRAIN_TASKS) cells held out as in-distribution eval set.
# Must match the split fraction used in 01_fit to keep the eval set identical.
EVAL_CELL_FRAC = 0.10

# Trial budgets to sweep.  "full" = all trials in the 90% training cells.
# Each integer budget B means: uniformly sample B trials (without replacement
# from the pool of all individual training trials; with replacement if B
# exceeds the pool size).
TRIAL_BUDGETS = [10, 100, 1_000, 10_000, 100_000, "full"]

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
N_EPOCHS         = 1000    # phase 1: mapper warmup
PATIENCE         = 100     # phase 1 early stopping
N_EPOCHS_PHASE2  = 3000   # phase 2: attnpool fine-tuning
PATIENCE_PHASE2  = 50
LR               = 1e-2
LR_ATTNPOOL      = 1e-5
N_MC             = 100
FREEZE_ENCODER   = False
MAPPER_HIDDEN    = None


RUN_TAG = "frozen" if FREEZE_ENCODER else "attnpool"

# ---------------------------------------------------------------------------
# Task split (identical to 01_fit)
# ---------------------------------------------------------------------------
# TRAIN_TASKS = [
#     "right", "transparent", "glossy", "large",
#     "left", "opaque", "matte", "small",
#     "right_and_transparent", "left_and_transparent",
#     "right_and_glossy",      "left_and_glossy",
#     "transparent_and_glossy",
#     "large_and_transparent", "large_and_glossy",
#     "right_and_transparent_and_glossy",
#     "left_and_transparent_and_glossy",
#     "large_and_transparent_and_glossy",
# ]
# VAL_TASKS = [
#     "right_and_large",
#     "left_and_large",
#     "right_and_large_and_glossy",
#     "right_and_large_and_transparent",
# ]


TRAIN_TASKS = [
    # simple
    "right", "small", "transparent", "matte"
]
VAL_TASKS = [
    # lr × sl conjunctions — never seen during training
    "left", "large", "opaque", "glossy"
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

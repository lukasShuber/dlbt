"""
Configuration for behavior run0 / 01_fit.

Training uses 90% of main-image × TRAIN_TASKS cells (the remaining 10% are
held out as an in-distribution eval set for early stopping).  Probe images
are used for stim/task/joint generalization evaluation only — never for
early stopping.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
METADATA    = "stimuli/imgs/metadata.jsonl"
CACHE_PATH  = "stimuli/imgs/clip_rn50_features_v2.pt"
RESOURCES   = Path(__file__).parent.parent / "resources"
BEHAVIOR_CSV = RESOURCES / "dlbt-behavior.csv"
RESULTS_DIR = Path(__file__).parent / "results"

# ---------------------------------------------------------------------------
# Data handling
# ---------------------------------------------------------------------------
SEED        = 42
N_SEEDS     = 1
SEEDS       = [42]

USE_TRIAL_KINDS = ("main", "probe")

# Quality-filtering (same as run0)
MIN_CATCH_PERF     = 1.0
MAIN_PERF_QUANTILE = 0.95

# Fraction of (main × TRAIN_TASKS) cells held out for in-distribution early
# stopping.  The remaining 1-EVAL_FRAC fraction is used for training.
EVAL_CELL_FRAC = 0.10

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
N_EPOCHS_PHASE1 = 1000
PATIENCE_PHASE1 = 100
N_EPOCHS_PHASE2 = 3000
PATIENCE_PHASE2 = 50
LR              = 1e-2
LR_ATTNPOOL     = 1e-5
N_MC            = 100
FREEZE_ENCODER       = True
MAPPER_HIDDEN        = None
THRESHOLD_CORRECTION = True   # run arity-adjusted τₙ MC inference after training


RUN_TAG = "frozen" if FREEZE_ENCODER else "attnpool"

# ---------------------------------------------------------------------------
# Task split (identical to run0)
# ---------------------------------------------------------------------------

TRAIN_TASKS = [
    # simple
    "right", "small", "transparent", "matte"
]
VAL_TASKS = [
    # lr × sl conjunctions — never seen during training

    "right_and_large",
    "left_and_glossy", "transparent_and_glossy",
    "left", "large", "opaque", "glossy",
    "left_and_large",
    "large_and_transparent", "large_and_glossy",
    "right_and_transparent", "left_and_transparent",
    "right_and_glossy",
    "right_and_transparent_and_glossy",
    "left_and_transparent_and_glossy",
    "large_and_transparent_and_glossy",
    "right_and_large_and_glossy",
    "right_and_large_and_transparent",
]

# ---------------------------------------------------------------------------
# Behavioural task_id → DLBT task name
# ---------------------------------------------------------------------------
BEH_ID_TO_TASK = {
    # 1-way
    "transparent":              "transparent",
    "opaque":                   "opaque",
    "glossy":                   "glossy",
    "matte":                    "matte",
    "large":                    "large",
    "small":                    "small",
    "right":                    "right",
    "left":                     "left",
    # 2-way (train)
    "transparent,right":        "right_and_transparent",
    "transparent,left":         "left_and_transparent",
    "glossy,right":             "right_and_glossy",
    "glossy,left":              "left_and_glossy",
    "transparent,glossy":       "transparent_and_glossy",
    "transparent,large":        "large_and_transparent",
    "glossy,large":             "large_and_glossy",
    # 3-way (train)
    "transparent,glossy,right": "right_and_transparent_and_glossy",
    "transparent,glossy,left":  "left_and_transparent_and_glossy",
    "transparent,glossy,large": "large_and_transparent_and_glossy",
    # val (lr × sl)
    "large,right":              "right_and_large",
    "large,left":               "left_and_large",
    "glossy,large,right":       "right_and_large_and_glossy",
    "transparent,large,right":  "right_and_large_and_transparent",
}

# ---------------------------------------------------------------------------
# Plot colours
# ---------------------------------------------------------------------------
C_TRAIN = "#E76F51"
C_EVAL  = "#F4A261"   # in-distribution eval (held-out cells, main images)
C_STIM  = "#457B9D"
C_TASK  = "#9B5DE5"
C_JOINT = "#43AA8B"

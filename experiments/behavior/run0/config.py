"""
Configuration for behavior run0 — pendant to simulation 01.

Same latent space (K=16, 4-bit) and same TRAIN/VAL task split as
experiments/simulations/01_four_dim_generalization — but the behavioural
counts come from real human data (dlbt-behavior.csv) rather than a
synthetic Dirichlet observer.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
METADATA    = "stimuli/imgs/metadata.jsonl"
CACHE_PATH  = "stimuli/imgs/clip_rn50_features_v2.pt"
RESOURCES   = Path(__file__).parent / "resources"
BEHAVIOR_CSV = RESOURCES / "dlbt-behavior.csv"
RESULTS_DIR = Path(__file__).parent / "results"

# ---------------------------------------------------------------------------
# Data handling
# ---------------------------------------------------------------------------
SEED          = 42
N_SEEDS       = 1
SEEDS         = [42]

# Image split: the behavioural experiment collects ~20 trials per (uid, task)
# on 16 probe images and ~3 trials per (uid, task) on ~980 main images.
# Probe images are used as the held-out eval set (stim_gen / joint_gen);
# main images are used for training.  This replaces the random fraction
# split used in simulation 01.

# Trial kinds used for training / evaluation.  `main` + `probe` are the
# scientific trials; `warmup` and `catch` are excluded.
USE_TRIAL_KINDS = ("main", "probe")

# Filtering criteria (from view-behavior.ipynb)
MIN_CATCH_PERF = 1.0   # must get 4/4 catch trials
MAIN_PERF_QUANTILE = 0.95   # above the 95th pctile of Binom(100, 0.5)

# ---------------------------------------------------------------------------
# Training (matches simulation 01)
# ---------------------------------------------------------------------------
N_EPOCHS_PHASE1 = 1000
PATIENCE_PHASE1 = 300
N_EPOCHS_PHASE2 = 3000
PATIENCE_PHASE2 = 100
LR              = 1e-2
LR_ATTNPOOL     = 1e-5
N_MC            = 1000
FREEZE_ENCODER  = True
MAPPER_HIDDEN   = None

RUN_TAG = "frozen" if FREEZE_ENCODER else "attnpool"

# ---------------------------------------------------------------------------
# Task split (identical to simulation 01)
# ---------------------------------------------------------------------------
TRAIN_TASKS = [
    # simple
    "right", "transparent", "glossy", "large",
    # 2-way
    "right_and_transparent",
    "right_and_glossy",
    "transparent_and_glossy",
    "large_and_transparent",
    "large_and_glossy",
    # 3-way
    "right_and_transparent_and_glossy",
    "large_and_transparent_and_glossy",
]
VAL_TASKS = [
    "right_and_large",
    "right_and_large_and_glossy",
    "right_and_large_and_transparent",
]

# ---------------------------------------------------------------------------
# Behavioural task_id  ->  DLBT task name
# ---------------------------------------------------------------------------
# task_id in the CSV is the comma-separated list of active factor values in
# fixed order:  transparency, glossiness, size, side.
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
# Plot colours (identical to simulation 01)
# ---------------------------------------------------------------------------
C_TRAIN = "#E76F51"
C_STIM  = "#457B9D"
C_TASK  = "#9B5DE5"
C_JOINT = "#43AA8B"

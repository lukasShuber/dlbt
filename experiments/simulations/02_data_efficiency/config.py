"""
Configuration for simulation 02 — data efficiency experiment.

Trains DLBT and SLDA across a range of *total* trial budgets.  For each
budget b, exactly b behavioral trials are sampled uniformly at random (with
replacement) from the full set of training (image, task) pairs.  At low
budgets most pairs are unobserved; at high budgets each pair accumulates
many trials.  This tests the full data-scarce → data-rich continuum.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths  (all relative to repo root)
# ---------------------------------------------------------------------------
METADATA    = "stimuli/imgs/metadata.jsonl"
CACHE_PATH  = "stimuli/imgs/clip_rn50_features_v2.pt"
RESULTS_DIR = Path(__file__).parent / "results"

# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------
BASE_SEED          = 42
N_SEEDS            = 1
SEEDS              = [BASE_SEED + i for i in range(N_SEEDS)]

BUDGETS    = [10, 100, 1_000, 10_000, 100_000, 1_000_000]  # total training trials
N_FULL_PER_PAIR = 1000   # trials per pair for the fixed test sets

PEAK               = 15.0
BASE_CONCENTRATION = 1.0
BETA               = 5.0
SCALE_BETA         = 10.0
IMG_TEST_FRAC      = 0.20

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
N_EPOCHS_PHASE1 = 1000
PATIENCE_PHASE1 = 100
N_EPOCHS_PHASE2 = 3000
PATIENCE_PHASE2 = 100
LR              = 1e-2
LR_ATTNPOOL     = 1e-5
N_MC            = 200
FREEZE_ENCODER  = True    # True → frozen only; False → phase 1 + attnpool fine-tune
MAPPER_HIDDEN   = None

RUN_TAG = "frozen" if FREEZE_ENCODER else "attnpool"

# ---------------------------------------------------------------------------
# Tasks  (identical to simulation 01)
# ---------------------------------------------------------------------------
TRAIN_TASKS = [
    # simple
    "left_right", "transparent", "glossy", "large",
    "left", "opaque", "matte", "small",
    # 2-way: lr × material
    "right_and_transparent", "left_and_transparent",
    "right_and_glossy",      "left_and_glossy",
    # 2-way: material × material
    "transparent_and_glossy",
    # 2-way: sl × material  (no lr × sl)
    "large_and_transparent", "large_and_glossy",
    # 3-way
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

# ---------------------------------------------------------------------------
# Plot colours
# ---------------------------------------------------------------------------
C_TRAIN = "#d95f02"
C_STIM  = "#1f78b4"
C_TASK  = "#7570b3"
C_JOINT = "#33a02c"

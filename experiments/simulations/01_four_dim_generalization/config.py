"""
Configuration for simulation 01 — four-dimensional generalization experiment.

Latent space: left/right × transparent × glossy × small/large  (K=16, 4-bit).
Holdout design: lr × sl conjunctions withheld as unseen tasks.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths  (all relative to repo root)
# ---------------------------------------------------------------------------
METADATA   = "stimuli/imgs/metadata.jsonl"
CACHE_PATH = "stimuli/imgs/clip_rn50_features_v2.pt"
RESULTS_DIR = Path(__file__).parent / "results"

# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------
SEED               = 42
N_TRIALS           = 100    # simulated decisions per (image, task)
PEAK               = 15.0   # peak Dirichlet concentration for clear images
BASE_CONCENTRATION = 1.0
BETA               = 5.0    # sigmoid sharpness for lr / transp / gloss dims
SCALE_BETA         = 10.0   # sigmoid sharpness for scale dim [0.2, 0.8]
IMG_TEST_FRAC      = 0.20   # fraction of images held out (stim / joint gen)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
N_EPOCHS_PHASE1 = 1000
PATIENCE_PHASE1 = 100
N_EPOCHS_PHASE2 = 3000
PATIENCE_PHASE2 = 100
LR              = 1e-2
LR_ATTNPOOL     = 1e-5
N_MC            = 200       # MC samples for choice_probs during training
FREEZE_ENCODER  = False     # True → frozen only; False → phase 1 + attnpool fine-tune
MAPPER_HIDDEN   = None      # None → linear mapper

RUN_TAG = "frozen" if FREEZE_ENCODER else "attnpool"

# ---------------------------------------------------------------------------
# Tasks
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
    # lr × sl conjunctions — never seen during training
    "right_and_large",
    "left_and_large",
    "right_and_large_and_glossy",
    "right_and_large_and_transparent",
]

# ---------------------------------------------------------------------------
# Plot colours  (consistent across all figures)
# ---------------------------------------------------------------------------
C_TRAIN = "#E76F51"   # coral   — training region
C_STIM  = "#457B9D"   # steel blue — stimulus generalization
C_TASK  = "#9B5DE5"   # electric purple — task generalization
C_JOINT = "#43AA8B"   # teal    — joint generalization

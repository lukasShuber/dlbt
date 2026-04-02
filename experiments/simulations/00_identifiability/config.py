"""
Configuration for simulation 00 — identifiability experiment.

Ground truth: alpha*(uid) = softplus(W* @ clip(uid) + b*)  — same functional
form as DLBT's mapper.  The target is therefore realizable: there exists a
set of DLBT parameters that perfectly reproduces the ground truth.

No train/test split.  No task holdout.  No SLDA comparison.
The single question: can DLBT recover W* from behavioral data?
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
METADATA    = "stimuli/imgs/metadata.jsonl"
CACHE_PATH  = "stimuli/imgs/clip_rn50_features_v2.pt"
RESULTS_DIR = Path(__file__).parent / "results"

# ---------------------------------------------------------------------------
# Ground-truth linear map  (fixed forever)
# ---------------------------------------------------------------------------
GT_SEED     = 0
ALPHA_SCALE = 4.0     # std of W* entries; std(logit) ≈ ALPHA_SCALE × ||feat||
                      # with ||feat|| ≈ 1.45 for CLIP RN50 → std(logit) ≈ 5.8

# ---------------------------------------------------------------------------
# Behavioral data
# ---------------------------------------------------------------------------
SEED     = 42
N_SEEDS  = 5
SEEDS    = [42, 43, 44, 45, 46]
N_TRIALS = 100    # decisions per (image, task)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
N_EPOCHS      = 1000
PATIENCE      = 150
LR            = 1e-2
N_MC          = 200
FREEZE_ENCODER = True   # linear mapper only — keeps the test clean
MAPPER_HIDDEN  = None

RUN_TAG = "frozen"

# ---------------------------------------------------------------------------
# Tasks  (all used — no holdout)
# ---------------------------------------------------------------------------
TASKS = [
    "right", "transparent", "glossy", "large",
    "left",  "opaque",      "matte",  "small",
    "right_and_transparent", "left_and_transparent",
    "right_and_glossy",      "left_and_glossy",
    "transparent_and_glossy",
    "large_and_transparent", "large_and_glossy",
    "right_and_transparent_and_glossy",
    "left_and_transparent_and_glossy",
    "large_and_transparent_and_glossy",
    "right_and_large",
    "left_and_large",
    "right_and_large_and_glossy",
    "right_and_large_and_transparent",
]

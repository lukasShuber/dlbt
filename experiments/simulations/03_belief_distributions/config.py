"""
Configuration for simulation 03 — belief distribution robustness.

Tests whether DLBT (which internally assumes a Dirichlet belief distribution)
remains accurate when the ground-truth observer uses a different distribution.

Four ground-truth observers are compared:
  dirichlet       — correctly specified baseline (current model)
  logistic_normal — beliefs = softmax(mu + Gaussian noise), matched variance
  lapse           — Dirichlet + random-choice lapse trials
  threshold       — SDT-style: noisy threshold crossing per latent dimension
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
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

N_TRIALS           = 100
PEAK               = 15.0
BASE_CONCENTRATION = 1.0
BETA               = 5.0
SCALE_BETA         = 10.0
IMG_TEST_FRAC      = 0.20

# ---------------------------------------------------------------------------
# Distribution-specific parameters
# ---------------------------------------------------------------------------
# Lapse: probability of a random (uninformative) trial
LAPSE_RATE = 0.15

# Threshold: Gaussian noise std on each logit (in logit space).
# sigma=1.5 gives roughly comparable difficulty to the Dirichlet baseline
# via the logistic-probit approximation (sigmoid ≈ Phi(x / 1.7)).
THRESHOLD_SIGMA = 1.5

# Distributions to sweep
DISTRIBUTIONS = ["dirichlet", "logistic_normal", "lapse", "threshold"]

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
FREEZE_ENCODER  = True   # False → run phase 2 attnpool fine-tuning
MAPPER_HIDDEN   = None
SLDA_GT         = False  # True → GT lstsq decoder; False → RidgeCV on behavioral data

RUN_TAG = "frozen" if FREEZE_ENCODER else "attnpool"

# ---------------------------------------------------------------------------
# Tasks  (identical to simulation 01)
# ---------------------------------------------------------------------------
TRAIN_TASKS = [
    "left_right", "transparent", "glossy", "large",
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

# ---------------------------------------------------------------------------
# Plot colours
# ---------------------------------------------------------------------------
C_TRAIN = "#E76F51"   # coral   — training region
C_STIM  = "#457B9D"   # steel blue — stimulus generalization
C_TASK  = "#9B5DE5"   # electric purple — task generalization
C_JOINT = "#43AA8B"   # teal    — joint generalization

DIST_COLORS = {
    "dirichlet":       "#555555",   # charcoal  — correctly-specified baseline
    "logistic_normal": "#E76F51",   # coral     — continuous-noise alternative
    "lapse":           "#43AA8B",   # teal      — lapse-rate alternative
    "threshold":       "#9B5DE5",   # purple    — threshold/SDT alternative
}
DIST_LABELS = {
    "dirichlet":       "Dirichlet",
    "logistic_normal": "Logistic-Normal",
    "lapse":           f"Lapse (λ={LAPSE_RATE})",
    "threshold":       f"Threshold (σ={THRESHOLD_SIGMA})",
}

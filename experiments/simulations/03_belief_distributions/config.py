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
METADATA    = "stimuli/imgs_pink/metadata.jsonl"
CACHE_PATH  = "stimuli/imgs_pink/clip_rn50_features_v2.pt"
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
# Training  (frozen encoder only — fast, apples-to-apples across distributions)
# ---------------------------------------------------------------------------
N_EPOCHS_PHASE1 = 1000
PATIENCE_PHASE1 = 100
LR              = 1e-2
N_MC            = 200
FREEZE_ENCODER  = True
MAPPER_HIDDEN   = None

RUN_TAG = "frozen"

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
C_TRAIN = "#d95f02"
C_STIM  = "#1f78b4"
C_TASK  = "#7570b3"
C_JOINT = "#33a02c"

DIST_COLORS = {
    "dirichlet":       "#333333",
    "logistic_normal": "#e7298a",
    "lapse":           "#1b9e77",
    "threshold":       "#d95f02",
}
DIST_LABELS = {
    "dirichlet":       "Dirichlet",
    "logistic_normal": "Logistic-Normal",
    "lapse":           f"Lapse (λ={LAPSE_RATE})",
    "threshold":       f"Threshold (σ={THRESHOLD_SIGMA})",
}

"""
Simulation 011 — emulate behavioural data constraints on synthetic agents.

Differences from sim 01:
  - Probe/main image asymmetry: N_TRIALS_MAIN (~3) vs N_TRIALS_PROBE (~20)
  - Per-dim sigmoid sharpness BETA_PER_DIM matching human accuracy hierarchy
  - Full 22-task behavioural task set (Option A — no held-out tasks)
"""

from pathlib import Path

METADATA    = "stimuli/imgs/metadata.jsonl"
CACHE_PATH  = "stimuli/imgs/clip_rn50_features_v2.pt"
RESULTS_DIR = Path(__file__).parent / "results"

# ---- Synthetic data ---------------------------------------------------------
SEED    = 42
N_SEEDS = 1
SEEDS   = [42]

# [011] Asymmetric trial counts matching behavioural collection
N_TRIALS_MAIN  = 3      # main images  (~3 trials / (uid, task) in behaviour)
N_TRIALS_PROBE = 20     # probe images (~20 trials / (uid, task) in behaviour)
N_PROBE_IMAGES = 16     # first N sorted UIDs treated as probe set

PEAK               = 15.0
BASE_CONCENTRATION = 1.0

# [011] Per-dim sigmoid sharpness — calibrated to match human 1-way accuracy:
#   lr=0.85  tr=0.73  gl=0.62  sl=0.76
BETA_PER_DIM = dict(
    lr = 5.0,
    tr = 5.0,
    gl = 5.0,
    sl = 5.0,
)

# ---- Training (mirrors sim 01) ---------------------------------------------
N_EPOCHS_PHASE1 = 1000
PATIENCE_PHASE1 = 100
N_EPOCHS_PHASE2 = 3000
PATIENCE_PHASE2 = 100
LR              = 1e-2
LR_ATTNPOOL     = 1e-5
N_MC            = 200
FREEZE_ENCODER  = True
MAPPER_HIDDEN   = None
RUN_TAG = "frozen" if FREEZE_ENCODER else "attnpool"

# ---- Tasks ------------------------
TRAIN_TASKS = [
    # 1-way only — perfectly balanced polarity per dim
    "right", "left",
    "transparent", "opaque",
    "glossy", "matte",
    "large", "small",
]
VAL_TASKS = [
    # held-out conjunctions — pure task generalisation test
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

# ---- Plot colours (consistent with other experiments) ----------------------
C_TRAIN = "#E76F51"
C_STIM  = "#457B9D"
C_TASK  = "#9B5DE5"
C_JOINT = "#43AA8B"

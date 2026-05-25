"""
Configuration for behavior run1 / 021_efficiency_main.

Key improvements over 02:
  - X-axis: trials per task  (total budget = tpt × n_tasks).
  - Budget grid: 10, 22, 46, 100, 215, 464, 1000 trials/task  (+  all data).
  - Per-budget 90/10 trial split for early stopping / model selection.
  - DLBT model selection: compare trained vs. base (concentration=1000,
    symmetric Dirichlet → P ≈ 0.5 everywhere under normalised utility).
  - SLDA: L2 logistic regression (C=1.0); per-task 90/10 split;
    compare fitted vs. base model (P=0.5) for each task.
  - Frozen CLIP encoder for both DLBT and SLDA (no attnpool fine-tuning).

Run from repo root:
    python experiments/behavior/run1/021_efficiency_main/run.py
    python experiments/behavior/run1/021_efficiency_main/analysis.py
"""

from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
METADATA    = "stimuli/imgs/metadata.jsonl"
CACHE_PATH  = "stimuli/imgs/clip_rn50_features_v2.pt"
RESULTS_DIR = Path(__file__).parent / "results"

_RUN1_DIR = Path(__file__).parent.parent
import importlib.util as _ilu
_spec     = _ilu.spec_from_file_location("_run1_cfg", _RUN1_DIR / "config.py")
_run1_cfg = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_run1_cfg)

BEHAVIOR_CSV_RUN0 = _run1_cfg.BEHAVIOR_CSV_RUN0
BEHAVIOR_CSV_RUN1 = _run1_cfg.BEHAVIOR_CSV_RUN1
BEH_ID_TO_TASK    = _run1_cfg.BEH_ID_TO_TASK

# ---------------------------------------------------------------------------
# Data handling
# ---------------------------------------------------------------------------
SEED                 = _run1_cfg.SEED
USE_TRIAL_KINDS      = _run1_cfg.USE_TRIAL_KINDS
MIN_CATCH_PERF       = _run1_cfg.MIN_CATCH_PERF
MAIN_PERF_QUANTILE   = _run1_cfg.MAIN_PERF_QUANTILE
MIN_TASK_ASSIGNMENTS = _run1_cfg.MIN_TASK_ASSIGNMENTS

# ---------------------------------------------------------------------------
# Budget grid  (trials per task)
# ---------------------------------------------------------------------------
# 3 log-spaced points per decade (decade start + 2 intermediates),
# spanning 10^1 to 10^3.  All-data is a separate special point in run.py.
TRIALS_PER_TASK: list[int] = sorted({
    int(round(10 ** (lo + k / 3)))
    for lo in range(1, 3)   # decades: 10^1, 10^2
    for k in range(3)       # k=0,1,2  →  start + 2 intermediates
} | {1_000})                # include 10^3 endpoint
# → [10, 22, 46, 100, 215, 464, 1000]

# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------
N_SEEDS = 5
SEEDS   = [42, 43, 44, 45, 46]

# ---------------------------------------------------------------------------
# Fast-pass mode  (quick smoke-test: smallest budget only)
# ---------------------------------------------------------------------------
FAST_PASS = False

# ---------------------------------------------------------------------------
# Encoder freeze flags
# ---------------------------------------------------------------------------
FREEZE_ENCODER_DLBT = False   # False → Phase 2 attnpool fine-tuning for DLBT
FREEZE_ENCODER_SLDA = True   # False → Phase 2 attnpool fine-tuning for SLDA

# ---------------------------------------------------------------------------
# Training — DLBT
# ---------------------------------------------------------------------------
N_EPOCHS          = 1000
PATIENCE          = 200
LR                = 0.01
N_MC              = 1000
NORMALIZED_UTILITY = True

# ---------------------------------------------------------------------------
# Training — Phase 2 (attnpool fine-tuning, DLBT and SLDA)
# ---------------------------------------------------------------------------
N_EPOCHS_PHASE2 = 3000
PATIENCE_PHASE2 = 50
LR_ATTNPOOL     = 1e-5

# ---------------------------------------------------------------------------
# DLBT base model
# ---------------------------------------------------------------------------
# Symmetric Dirichlet with very high concentration → beliefs locked at 1/K.
# Under normalised utility the SEU score is exactly 0 → P(right) = 0.5.
# Used as the comparison point for per-budget model selection.
BASE_CONCENTRATION = 1000.0

# ---------------------------------------------------------------------------
# SLDA — L2 logistic regression with cross-validated regularisation
# ---------------------------------------------------------------------------
SLDA_Cs       = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]   # C grid for CV
SLDA_MAX_ITER = 1000

# ---------------------------------------------------------------------------
# Mapper initialisation
# ---------------------------------------------------------------------------
INIT_MODE       = "random"
INIT_ALPHA_LOW  = 0.6
INIT_ALPHA_HIGH = 0.7

# ---------------------------------------------------------------------------
# Run tag  (encodes encoder settings so different runs don't overwrite each other)
# ---------------------------------------------------------------------------
_enc_dlbt = "frozen" if FREEZE_ENCODER_DLBT else "attnpool"
_enc_slda = "frozen" if FREEZE_ENCODER_SLDA else "attnpool"
RUN_TAG   = f"efficiency_main_021_dlbt_{_enc_dlbt}_slda_{_enc_slda}"

# ---------------------------------------------------------------------------
# Plot options
# ---------------------------------------------------------------------------
LOG_Y = True

# ---------------------------------------------------------------------------
# Plot colours
# ---------------------------------------------------------------------------
C_DLBT   = "#C0392B"   # DLBT — saturated red
C_SLDA   = "#7D3C98"   # SLDA — purple
C_ANTI   = "#777777"   # anti-human DLBT — medium gray
C_RNDINI = "#999999"   # reference lines

ARITY_COLOR = {1: "#2a6fb5", 2: "#43AA8B", 3: "#E76F51", 4: "#9B5DE5"}


# ---------------------------------------------------------------------------
# Helper: eligible tasks
# ---------------------------------------------------------------------------
def eligible_tasks(df_filtered):
    tasks = _run1_cfg.eligible_tasks(df_filtered, MIN_TASK_ASSIGNMENTS)
    return sorted(tasks, key=lambda t: (t.count("_and_"), t))

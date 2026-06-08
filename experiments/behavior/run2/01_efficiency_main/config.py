"""
Configuration for behavior run2 / 01_efficiency_main.

Full dataset: run0 + run1 + run2 (80 tasks, frozen CLIP encoders for both
DLBT and SLDA — no attnpool fine-tuning).

Budget grid: 10, 22, 46, 100, 215, 464, 1000 trials/task  (+  all data).

Run from repo root:
    python experiments/behavior/run2/01_efficiency_main/run.py
    python experiments/behavior/run2/01_efficiency_main/analysis.py
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
METADATA    = "stimuli/imgs/metadata.jsonl"
CACHE_PATH  = "stimuli/imgs/clip_rn50_features_v2.pt"
RESULTS_DIR = Path(__file__).parent / "results"

_RUN2_DIR = Path(__file__).parent.parent
import importlib.util as _ilu
_spec     = _ilu.spec_from_file_location("_run2_cfg", _RUN2_DIR / "config.py")
_run2_cfg = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_run2_cfg)

BEHAVIOR_CSV_RUN0 = _run2_cfg.BEHAVIOR_CSV_RUN0
BEHAVIOR_CSV_RUN1 = _run2_cfg.BEHAVIOR_CSV_RUN1
BEHAVIOR_CSV_RUN2 = _run2_cfg.BEHAVIOR_CSV_RUN2
BEH_ID_TO_TASK    = _run2_cfg.BEH_ID_TO_TASK

# ---------------------------------------------------------------------------
# Data handling
# ---------------------------------------------------------------------------
SEED                 = _run2_cfg.SEED
USE_TRIAL_KINDS      = _run2_cfg.USE_TRIAL_KINDS
MIN_CATCH_PERF       = _run2_cfg.MIN_CATCH_PERF
MAIN_PERF_QUANTILE   = _run2_cfg.MAIN_PERF_QUANTILE
MIN_TASK_ASSIGNMENTS = _run2_cfg.MIN_TASK_ASSIGNMENTS

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
N_SEEDS = 3
SEEDS   = [42, 43, 44]

# ---------------------------------------------------------------------------
# Fast-pass mode  (quick smoke-test: smallest budget only)
# ---------------------------------------------------------------------------
FAST_PASS = False

# ---------------------------------------------------------------------------
# Encoder freeze flags
# ---------------------------------------------------------------------------
FREEZE_ENCODER_DLBT = False    # True → frozen CLIP, mapper only
FREEZE_ENCODER_SLDA = False    # True → frozen CLIP, logistic regression only

# ---------------------------------------------------------------------------
# Training — DLBT
# ---------------------------------------------------------------------------
N_EPOCHS           = 1000
PATIENCE           = 200
LR                 = 0.01
N_MC               = 1000
NORMALIZED_UTILITY = True

# ---------------------------------------------------------------------------
# Training — Phase 2 (attnpool fine-tuning, unused when encoders are frozen)
# ---------------------------------------------------------------------------
N_EPOCHS_PHASE2  = 3000
PATIENCE_PHASE2  = 50
LR_ATTNPOOL_DLBT = 1e-5
LR_ATTNPOOL_SLDA = 1e-6

# ---------------------------------------------------------------------------
# DLBT base model
# ---------------------------------------------------------------------------
# Symmetric Dirichlet with very high concentration → beliefs locked at 1/K.
# Under normalised utility the SEU score is exactly 0 → P(right) = 0.5.
BASE_CONCENTRATION = 1000.0

# ---------------------------------------------------------------------------
# SLDA — L2 logistic regression with cross-validated regularisation
# ---------------------------------------------------------------------------
SLDA_Cs       = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
SLDA_MAX_ITER = 1000

# ---------------------------------------------------------------------------
# Mapper initialisation
# ---------------------------------------------------------------------------
INIT_MODE       = "random"
INIT_ALPHA_LOW  = 0.6
INIT_ALPHA_HIGH = 0.7

# ---------------------------------------------------------------------------
# Run tag
# ---------------------------------------------------------------------------
_enc_dlbt = "frozen" if FREEZE_ENCODER_DLBT else "attnpool"
_enc_slda = "frozen" if FREEZE_ENCODER_SLDA else "attnpool"
RUN_TAG   = f"efficiency_main_01_dlbt_{_enc_dlbt}_slda_{_enc_slda}"

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
    tasks = _run2_cfg.eligible_tasks(df_filtered, MIN_TASK_ASSIGNMENTS)
    return sorted(tasks, key=lambda t: (t.count("_and_"), t))

"""
Configuration for behavior run1 / 023_efficiency_main.

Single-condition (full task coverage) budget sweep with:
  - Bootstrap fallback sampling (sampling with replacement only when the
    per-task budget exceeds the available pool for that task).
  - Anti-human DLBT reference: DLBT trained on label-flipped data.
  - Dense log-spaced budget grid (2 intermediate points per decade).
  - Separate "all-data" point (all pool trials, no sampling) plotted as a
    disconnected filled marker.

Run from repo root:
    python experiments/behavior/run1/023_efficiency_main/run.py
    python experiments/behavior/run1/023_efficiency_main/analysis.py
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
# Budget grid
# ---------------------------------------------------------------------------
# 3 log-spaced points per decade (decade start + 2 intermediates),
# spanning 10^2 to 10^5.  "All data" is a separate special point in run.py.
TRIAL_BUDGETS: list[int] = sorted({
    int(round(10 ** (lo + k / 3)))
    for lo in range(2, 5)   # decades: 10^2, 10^3, 10^4
    for k in range(3)       # k=0,1,2  →  start + 2 intermediates
} | {100_000})              # include 10^5 endpoint

# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------
N_SEEDS = 1
SEEDS   = [42]

# ---------------------------------------------------------------------------
# Fast-pass mode  (quick smoke-test: min + max budget only)
# ---------------------------------------------------------------------------
# Set FAST_PASS = True to run only two budget points: the smallest grid point
# and total_pool_size (the complete training dataset).  Intermediate grid
# points are skipped.  The separate all-data section always runs regardless.
# Useful for a quick end-to-end sanity check before committing to a full sweep.
FAST_PASS = True

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
N_EPOCHS        = 1000
PATIENCE        = 200
N_EPOCHS_PHASE2 = 3000
PATIENCE_PHASE2 = 50
LR              = 0.01
LR_ATTNPOOL     = 1e-5
N_MC            = 1000
FREEZE_ENCODER     = True
MAPPER_HIDDEN      = None
NORMALIZED_UTILITY = True

# ---------------------------------------------------------------------------
# Mapper initialisation
# ---------------------------------------------------------------------------
INIT_MODE       = "random"
INIT_ALPHA      = 1.0
INIT_ALPHA_LOW  = 0.6
INIT_ALPHA_HIGH = 0.7
INIT_SEED       = 0

# ---------------------------------------------------------------------------
# Median threshold correction (off by default; set True to enable)
# ---------------------------------------------------------------------------
MEDIAN_CORRECTION = False
NEUTRAL_ALPHA     = (INIT_ALPHA_LOW + INIT_ALPHA_HIGH) / 2   # 0.65

RUN_TAG = "efficiency_main"

# ---------------------------------------------------------------------------
# Plot options
# ---------------------------------------------------------------------------
LOG_Y = False   # log-scale y-axis on the cMSE figure (also overridable via --log-y)

# ---------------------------------------------------------------------------
# Plot colours
# ---------------------------------------------------------------------------
C_DLBT   = "#C0392B"   # DLBT — saturated red
C_SLDA   = "#7D3C98"   # SLDA — purple
C_ANTI   = "#777777"   # anti-human DLBT — medium gray (solid; distinct from ref lines)
C_RNDINI = "#999999"   # reference lines (random guesser / random-init DLBT)

ARITY_COLOR = {1: "#2a6fb5", 2: "#43AA8B", 3: "#E76F51", 4: "#9B5DE5"}


# ---------------------------------------------------------------------------
# Helper: eligible tasks
# ---------------------------------------------------------------------------
def eligible_tasks(df_filtered):
    tasks = _run1_cfg.eligible_tasks(df_filtered, MIN_TASK_ASSIGNMENTS)
    return sorted(tasks, key=lambda t: (t.count("_and_"), t))

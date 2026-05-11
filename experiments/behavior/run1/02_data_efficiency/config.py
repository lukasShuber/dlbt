"""
Configuration for behavior run1 / 02_data_efficiency — coverage sweep.

DLBT is trained on cumulative-nested random task subsets of varying coverage;
SLDA is trained on all tasks as a reference baseline.  Both are evaluated on
the full 80-task probe matrix (16 probe images × 80 tasks).

COVERAGE_FRACS : fractions of all eligible tasks used for DLBT training.
                 Subsets are cumulative-nested per seed:
                 10 % ⊂ 25 % ⊂ 50 % ⊂ 75 % ⊂ 100 %.
N_SEEDS        : random task orderings → enables SEM shading in plots.
                 Start with 1; bump up when results are ready.
TRIAL_BUDGETS  : fixed series [10, 100, …, 100_000].  Each trace starts at
                 min_budget = n_train_tasks (q=1) and ends at
                 max_budget = min_pool_size × n_train_tasks (no repeats).
                 Budget points outside [min, max] are dropped automatically.
"""

from pathlib import Path
import sys

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

BEHAVIOR_CSV_RUN0    = _run1_cfg.BEHAVIOR_CSV_RUN0
BEHAVIOR_CSV_RUN1    = _run1_cfg.BEHAVIOR_CSV_RUN1
BEH_ID_TO_TASK       = _run1_cfg.BEH_ID_TO_TASK

# ---------------------------------------------------------------------------
# Data handling
# ---------------------------------------------------------------------------
SEED                 = _run1_cfg.SEED
USE_TRIAL_KINDS      = _run1_cfg.USE_TRIAL_KINDS
MIN_CATCH_PERF       = _run1_cfg.MIN_CATCH_PERF
MAIN_PERF_QUANTILE   = _run1_cfg.MAIN_PERF_QUANTILE
MIN_TASK_ASSIGNMENTS = _run1_cfg.MIN_TASK_ASSIGNMENTS

# ---------------------------------------------------------------------------
# Coverage sweep
# ---------------------------------------------------------------------------
COVERAGE_FRACS = [0.10, 0.25, 0.50, 0.75, 1.00]

N_SEEDS = 1       # number of random task orderings; bump for SEM bands
SEEDS   = [42]    # one seed to start

# Fixed trial budget series.  Per-trace start/end is computed dynamically.
TRIAL_BUDGETS = [10, 100, 1_000, 10_000, 100_000]

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

RUN_TAG = ("frozen" if FREEZE_ENCODER else "attnpool") + "_coverage_norm"

# ---------------------------------------------------------------------------
# Plot colours (used in learning-curve plots)
# ---------------------------------------------------------------------------
C_TRAIN = "#E76F51"
C_EVAL  = "#F4A261"


# ---------------------------------------------------------------------------
# Helper: eligible tasks
# ---------------------------------------------------------------------------
def eligible_tasks(df_filtered):
    """Return list of DLBT task names sorted by (arity, name) with sufficient data."""
    tasks = _run1_cfg.eligible_tasks(df_filtered, MIN_TASK_ASSIGNMENTS)
    return sorted(tasks, key=lambda t: (t.count("_and_"), t))

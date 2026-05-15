"""
Configuration for behavior run1 / 05_determ_beliefs — deterministic-belief ablation.

DetBTAgent replaces the Monte Carlo Dirichlet integration of DlbtAgent with a
deterministic forward pass through the Dirichlet mean:

    μ_x  = α_x / Σ_k α_xk        (point mass at Dirichlet mean)
    p̃_xt = σ(μ_x · ΔU_t)

The same two sweep axes as 02 and 022 are covered in one run:
  - Coverage sweep  (coverage fracs [10 %, 25 %, 50 %, 75 %, 100 %])
  - Arity sweep     (arities [1, 2, 3, 4] with equal n_tasks)

SLDA is always trained on all eligible tasks as a fixed reference baseline.
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

# ---------------------------------------------------------------------------
# Arity sweep
# ---------------------------------------------------------------------------
ARITIES = [1, 2, 3, 4]

# Number of tasks to train on per arity.
# None → computed dynamically in run.py as min(n_eligible_tasks per arity).
N_TASKS_PER_ARITY = None

# ---------------------------------------------------------------------------
# Shared sweep settings
# ---------------------------------------------------------------------------
N_SEEDS = 1
SEEDS   = [42]

# Fixed trial budget series.  Per-trace start/end computed dynamically.
TRIAL_BUDGETS = [10, 100, 1_000, 10_000, 100_000]

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
N_EPOCHS    = 1000
PATIENCE    = 200
LR          = 0.01

# DetBTAgent is always trained with frozen encoder (no attnpool phase 2).
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

RUN_TAG = "det_beliefs_norm"

# ---------------------------------------------------------------------------
# Plot colours  (mirror 02 and 022 conventions)
# ---------------------------------------------------------------------------
import matplotlib.pyplot as _plt
_BLUES = _plt.get_cmap("Blues")
_CMAP_OFFSETS = {0.10: 0.30, 0.25: 0.44, 0.50: 0.58, 0.75: 0.72, 1.00: 0.88}
def cov_color(frac: float):
    return _BLUES(_CMAP_OFFSETS.get(frac, 0.6))

ARITY_COLOR = {1: "#2a6fb5", 2: "#43AA8B", 3: "#E76F51", 4: "#9B5DE5"}
C_SLDA      = "#7D3C98"
C_TRAIN     = "#E76F51"
C_EVAL      = "#F4A261"


# ---------------------------------------------------------------------------
# Helpers: eligible tasks, arity utilities
# ---------------------------------------------------------------------------
def eligible_tasks(df_filtered):
    """Return list of DLBT task names sorted by (arity, name) with sufficient data."""
    tasks = _run1_cfg.eligible_tasks(df_filtered, MIN_TASK_ASSIGNMENTS)
    return sorted(tasks, key=lambda t: (t.count("_and_"), t))


def arity_of(task_name: str) -> int:
    return task_name.count("_and_") + 1


def tasks_by_arity(task_list: list) -> dict[int, list]:
    """Split a task list into {arity: [tasks]} dict."""
    result = {a: [] for a in ARITIES}
    for t in task_list:
        a = arity_of(t)
        if a in result:
            result[a].append(t)
    return result

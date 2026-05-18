"""
Configuration for behavior run1 / 022_data_efficiency_arity — arity sweep.

DLBT is trained on N_TASKS_PER_ARITY randomly selected tasks from each arity
class [1, 2, 3, 4] and evaluated on the full 80-task probe matrix.

N_TASKS_PER_ARITY is fixed to the number of eligible 1-way tasks (the minimum
across arities), so training volume (n_tasks × trials/task) is identical across
arities. Randomness in which tasks are selected is averaged over N_SEEDS.

SLDA is trained on all tasks as a reference baseline.
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
# Arity sweep
# ---------------------------------------------------------------------------
ARITIES = [1, 2, 3, 4]

# Number of tasks to train on per arity.
# None → computed dynamically in run.py as min(n_eligible_tasks per arity).
N_TASKS_PER_ARITY = None

N_SEEDS = 5
SEEDS   = [42, 43, 44, 45, 46]

# Fixed trial budget series — same logic as 02_data_efficiency.
TRIAL_BUDGETS = [10, 100, 1_000, 10_000, 100_000]

# ---------------------------------------------------------------------------
# Training  (mirrors 02_data_efficiency)
# ---------------------------------------------------------------------------
N_EPOCHS        = 1000
PATIENCE        = 200
N_EPOCHS_PHASE2 = 3000
PATIENCE_PHASE2 = 50
LR              = 0.01
LR_ATTNPOOL     = 1e-5
N_MC            = 1000
FREEZE_ENCODER      = False  # DLBT: freeze CLIP encoder (no attnpool fine-tuning)
FREEZE_ENCODER_SLDA = True   # SLDA: freeze CLIP encoder (no attnpool fine-tuning)
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
# Median threshold correction
# ---------------------------------------------------------------------------
MEDIAN_CORRECTION = False
NEUTRAL_ALPHA     = (INIT_ALPHA_LOW + INIT_ALPHA_HIGH) / 2   # 0.65

RUN_TAG = ("frozen" if FREEZE_ENCODER else "attnpool") + "_arity_norm"

# ---------------------------------------------------------------------------
# Plot colours
# ---------------------------------------------------------------------------
ARITY_COLOR = {1: "#2a6fb5", 2: "#43AA8B", 3: "#E76F51", 4: "#9B5DE5"}
C_SLDA      = "#7D3C98"
C_TRAIN     = "#E76F51"
C_EVAL      = "#F4A261"


# ---------------------------------------------------------------------------
# Helper: eligible tasks
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

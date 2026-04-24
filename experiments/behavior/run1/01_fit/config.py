"""
Configuration for behavior run1 / 01_fit.

Task split: seeded 80/20 random split over the eligible tasks
(those with >= MIN_TASK_ASSIGNMENTS filtered assignments).
TRAIN_TASKS / VAL_TASKS are computed once at import time and are
fully reproducible given SPLIT_SEED and MIN_TASK_ASSIGNMENTS.
"""

from pathlib import Path
import sys

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
METADATA         = "stimuli/imgs/metadata.jsonl"
CACHE_PATH       = "stimuli/imgs/clip_rn50_features_v2.pt"
RESULTS_DIR      = Path(__file__).parent / "results"

_RUN1_DIR = Path(__file__).parent.parent
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("_run1_cfg", _RUN1_DIR / "config.py")
_run1_cfg = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_run1_cfg)

BEHAVIOR_CSV_RUN0 = _run1_cfg.BEHAVIOR_CSV_RUN0
BEHAVIOR_CSV_RUN1 = _run1_cfg.BEHAVIOR_CSV_RUN1
BEH_ID_TO_TASK    = _run1_cfg.BEH_ID_TO_TASK

# ---------------------------------------------------------------------------
# Data handling
# ---------------------------------------------------------------------------
SEED               = _run1_cfg.SEED
N_SEEDS            = 1
SEEDS              = [42]

USE_TRIAL_KINDS    = _run1_cfg.USE_TRIAL_KINDS
MIN_CATCH_PERF     = _run1_cfg.MIN_CATCH_PERF
MAIN_PERF_QUANTILE = _run1_cfg.MAIN_PERF_QUANTILE
MIN_TASK_ASSIGNMENTS = _run1_cfg.MIN_TASK_ASSIGNMENTS

EVAL_CELL_FRAC = 0.10

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
N_EPOCHS_PHASE1  = 1000
PATIENCE_PHASE1  = 100
N_EPOCHS_PHASE2  = 3000
PATIENCE_PHASE2  = 50
LR               = 1e-2
LR_ATTNPOOL      = 1e-5
N_MC             = 100
FREEZE_ENCODER   = False
MAPPER_HIDDEN    = None

RUN_TAG = "frozen" if FREEZE_ENCODER else "attnpool"

# ---------------------------------------------------------------------------
# 80/20 random task split — computed once at import from eligible tasks
# ---------------------------------------------------------------------------
TRAIN_FRAC = 0.80
SPLIT_SEED = 0    # separate from data SEED so the two are independent

def _compute_split():
    import numpy as np
    import pandas as pd
    sys.path.insert(0, str(_RUN1_DIR.parent / "run0"))
    from preprocess import filter_assignments

    df_raw = pd.concat(
        [pd.read_csv(BEHAVIOR_CSV_RUN0),
         pd.read_csv(BEHAVIOR_CSV_RUN1)],
        ignore_index=True,
    )
    df_f, _ = filter_assignments(
        df_raw,
        min_catch_perf     = MIN_CATCH_PERF,
        main_perf_quantile = MAIN_PERF_QUANTILE,
        seed               = SEED,
    )
    tasks = sorted(_run1_cfg.eligible_tasks(df_f, MIN_TASK_ASSIGNMENTS))
    rng   = np.random.default_rng(SPLIT_SEED)
    idx   = rng.permutation(len(tasks))
    n_train = int(round(len(tasks) * TRAIN_FRAC))
    train = sorted(tasks[i] for i in idx[:n_train])
    val   = sorted(tasks[i] for i in idx[n_train:])
    return train, val

TRAIN_TASKS, VAL_TASKS = _compute_split()

# ---------------------------------------------------------------------------
# Plot colours
# ---------------------------------------------------------------------------
C_TRAIN = "#E76F51"
C_EVAL  = "#F4A261"
C_STIM  = "#457B9D"
C_TASK  = "#9B5DE5"
C_JOINT = "#43AA8B"

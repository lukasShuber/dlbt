"""
Configuration for behavior run1 / 02_data_efficiency.

Identical task split and model settings as 01_fit.  Training budgets are
varied from 10 to the full training set.

SPLIT_MODE controls the task split (mirrors run1/01_fit/config.py):
  "arity"  — TRAIN: all eligible 1-way tasks; VAL: all 2/3/4-way tasks.
  "random" — seeded 80/20 random split over all eligible tasks.
  "manual" — explicit MANUAL_TRAIN_TASKS / MANUAL_VAL_TASKS lists below.
             Both lists are intersected with eligible_tasks() so that
             MIN_TASK_ASSIGNMENTS is still respected.

THRESHOLD_CORRECTION = True runs arity-adjusted h_n MC inference after
each budget (in addition to the standard h=0 predictions).
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
SEED               = _run1_cfg.SEED
N_SEEDS            = 1
SEEDS              = [42]

USE_TRIAL_KINDS      = _run1_cfg.USE_TRIAL_KINDS
MIN_CATCH_PERF       = _run1_cfg.MIN_CATCH_PERF
MAIN_PERF_QUANTILE   = _run1_cfg.MAIN_PERF_QUANTILE
MIN_TASK_ASSIGNMENTS = _run1_cfg.MIN_TASK_ASSIGNMENTS

# 10% of (main × TRAIN_TASKS) cells held out as in-distribution eval set.
# Must match the split fraction used in 01_fit to keep the eval set identical.
EVAL_CELL_FRAC = 0.10

# Trial budgets to sweep.  "full" = all trials in the 90% training cells.
# Each integer B means: uniformly sample B trials (without replacement from
# the pool of all individual training trials; with replacement if B > pool).
TRIAL_BUDGETS = [10, 100, 1_000, 10_000, "full"]

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
N_EPOCHS        = 1000
PATIENCE        = 100
N_EPOCHS_PHASE2 = 3000
PATIENCE_PHASE2 = 50
LR              = 1e-2
LR_ATTNPOOL     = 1e-5
N_MC            = 100
FREEZE_ENCODER  = True
MAPPER_HIDDEN   = None

# Run arity-adjusted h_n MC inference after each budget (in addition to h=0).
THRESHOLD_CORRECTION = False

# ---------------------------------------------------------------------------
# Task split
# ---------------------------------------------------------------------------
SPLIT_MODE = "random"   # "arity" | "random" | "manual"
SPLIT_SEED = 0         # used only when SPLIT_MODE == "random"
TRAIN_FRAC = 0.80      # used only when SPLIT_MODE == "random"

# Used only when SPLIT_MODE == "manual".
# Both lists are intersected with eligible_tasks() at import time.
MANUAL_TRAIN_TASKS: list = [
    # example — replace with your desired training tasks:
    # "right", "small", "transparent", "matte",
]
MANUAL_VAL_TASKS: list = [
    # example — replace with your desired held-out tasks:
    # "right_and_large", "left_and_glossy",
]

RUN_TAG = ("frozen" if FREEZE_ENCODER else "attnpool") + f"_{SPLIT_MODE}"


def _compute_split():
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
    all_eligible = sorted(_run1_cfg.eligible_tasks(df_f, MIN_TASK_ASSIGNMENTS))

    if SPLIT_MODE == "arity":
        train = sorted(t for t in all_eligible if "_and_" not in t)
        val   = sorted(t for t in all_eligible if "_and_"     in t)

    elif SPLIT_MODE == "random":
        import numpy as np
        rng     = np.random.default_rng(SPLIT_SEED)
        idx     = rng.permutation(len(all_eligible))
        n_train = int(round(len(all_eligible) * TRAIN_FRAC))
        train   = sorted(all_eligible[i] for i in idx[:n_train])
        val     = sorted(all_eligible[i] for i in idx[n_train:])

    elif SPLIT_MODE == "manual":
        eligible_set = set(all_eligible)
        train = sorted(t for t in MANUAL_TRAIN_TASKS if t in eligible_set)
        val   = sorted(t for t in MANUAL_VAL_TASKS   if t in eligible_set)
        dropped_train = [t for t in MANUAL_TRAIN_TASKS if t not in eligible_set]
        dropped_val   = [t for t in MANUAL_VAL_TASKS   if t not in eligible_set]
        if dropped_train:
            print(f"[config] manual split: dropped {len(dropped_train)} train tasks "
                  f"below MIN_TASK_ASSIGNMENTS: {dropped_train}")
        if dropped_val:
            print(f"[config] manual split: dropped {len(dropped_val)} val tasks "
                  f"below MIN_TASK_ASSIGNMENTS: {dropped_val}")

    else:
        raise ValueError(
            f"Unknown SPLIT_MODE {SPLIT_MODE!r}. Choose 'arity', 'random', or 'manual'."
        )

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

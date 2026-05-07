"""
Configuration for behavior run1 / 01_fit.

Task split controlled by SPLIT_MODE:
  "arity"  — TRAIN: all eligible 1-way tasks; VAL: all 2/3/4-way tasks.
             Tests generalisation from atomic dimensions to conjunctions.
  "random" — seeded 80/20 random split over all eligible tasks.
             Standard held-out evaluation across all arities.
  "manual" — explicit MANUAL_TRAIN_TASKS / MANUAL_VAL_TASKS lists below.
             Both lists are intersected with eligible_tasks() so that
             MIN_TASK_ASSIGNMENTS is still respected.
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
N_EPOCHS_PHASE1  = 2000
PATIENCE_PHASE1  = 200
N_EPOCHS_PHASE2  = 3000
PATIENCE_PHASE2  = 50
LR               = 0.01
LR_ATTNPOOL      = 1e-4
N_MC             = 1000
FREEZE_ENCODER     = True
MAPPER_HIDDEN      = None
NORMALIZED_UTILITY = True

# ---------------------------------------------------------------------------
# Mapper initialisation
# ---------------------------------------------------------------------------
# INIT_MODE = "uniform" — mapper bias set so softplus output starts at INIT_ALPHA
# INIT_MODE = "random"  — mapper bias drawn so output ~ U(INIT_ALPHA_LOW, INIT_ALPHA_HIGH)
INIT_MODE       = "random"
INIT_ALPHA      = 1.0
INIT_ALPHA_LOW  = 0.6
INIT_ALPHA_HIGH = 0.7
INIT_SEED       = 0

# ---------------------------------------------------------------------------
# Task split
# ---------------------------------------------------------------------------
SPLIT_MODE = "arity"      # "all" | "arity" | "random" | "manual"

# Used when SPLIT_MODE == "arity":
TRAIN_ARITIES = [4]         # arities included in training
HOLD_OUT_REST = True        # True → remaining arities go to val; False → no val

# Used when SPLIT_MODE == "random":
SPLIT_SEED = 0
TRAIN_FRAC = 0.80

# Used only when SPLIT_MODE == "manual".
# Both lists are intersected with eligible_tasks() at import time.
MANUAL_TRAIN_TASKS: list = [

"left_and_large_and_opaque_and_glossy", "left_and_large_and_opaque_and_matte",
"left_and_large_and_transparent_and_glossy", "left_and_large_and_transparent_and_matte",
"left_and_small_and_opaque_and_glossy", "left_and_small_and_opaque_and_matte",
"left_and_small_and_transparent_and_glossy", "left_and_small_and_transparent_and_matte",
"right_and_large_and_opaque_and_glossy", "right_and_large_and_opaque_and_matte",
"right_and_large_and_transparent_and_glossy", "right_and_large_and_transparent_and_matte",
"right_and_small_and_opaque_and_glossy", "right_and_small_and_opaque_and_matte",
"right_and_small_and_transparent_and_glossy", "right_and_small_and_transparent_and_matte"

]
MANUAL_VAL_TASKS: list = [
"glossy", "large", "left", "matte", "opaque", "right", "small", "transparent",

   "large_and_glossy", "large_and_matte", "large_and_opaque", "large_and_transparent",
"left_and_glossy", "left_and_large", "left_and_matte", "left_and_opaque", "left_and_small",
"left_and_transparent", "opaque_and_glossy", "opaque_and_matte", "right_and_glossy",
"right_and_large", "right_and_matte", "right_and_opaque", "right_and_small",
"right_and_transparent", "small_and_glossy", "small_and_matte", "small_and_opaque",
"small_and_transparent", "transparent_and_glossy", "transparent_and_matte"

"large_and_opaque_and_glossy", "large_and_opaque_and_matte",
"large_and_transparent_and_glossy", "large_and_transparent_and_matte",
"left_and_large_and_glossy", "left_and_large_and_matte", "left_and_large_and_opaque",
"left_and_large_and_transparent", "left_and_opaque_and_glossy", "left_and_opaque_and_matte",
"left_and_small_and_glossy", "left_and_small_and_matte", "left_and_small_and_opaque",
"left_and_small_and_transparent", "left_and_transparent_and_glossy",
"left_and_transparent_and_matte", "right_and_large_and_glossy", "right_and_large_and_matte",
"right_and_large_and_opaque", "right_and_large_and_transparent",
"right_and_opaque_and_glossy", "right_and_opaque_and_matte", "right_and_small_and_glossy",
"right_and_small_and_matte", "right_and_small_and_opaque", "right_and_small_and_transparent",
"right_and_transparent_and_glossy", "right_and_transparent_and_matte",
"small_and_opaque_and_glossy", "small_and_opaque_and_matte",
"small_and_transparent_and_glossy", "small_and_transparent_and_matte",
]

_nu_tag = "norm" if NORMALIZED_UTILITY else "raw"
RUN_TAG = ("frozen" if FREEZE_ENCODER else "attnpool") + f"_{SPLIT_MODE}_{_nu_tag}"

# ---------------------------------------------------------------------------
# Task split — computed once at import from eligible tasks
# ---------------------------------------------------------------------------

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

    if SPLIT_MODE == "all":
        return all_eligible, []

    elif SPLIT_MODE == "arity":
        def _arity(name): return name.count("_and_") + 1
        train = sorted(t for t in all_eligible if _arity(t) in TRAIN_ARITIES)
        val   = (sorted(t for t in all_eligible if _arity(t) not in TRAIN_ARITIES)
                 if HOLD_OUT_REST else [])

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
            f"Unknown SPLIT_MODE {SPLIT_MODE!r}. "
            f"Choose 'all', 'arity', 'random', or 'manual'."
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

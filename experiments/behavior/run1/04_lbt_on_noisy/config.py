"""
Configuration for behavior run1 / 04_lbt_on_noisy.

Like 03_lbt but trains on ALL images (probe + main), not just probe images.
Main images contribute noisier behavioral signal (fewer dedicated trials),
hence the name.  One Dirichlet α vector is learned per image UID.

SPLIT_MODE controls the train / val task split:

  "all"    — train on all eligible tasks, no val.

  "arity"  — train on tasks whose arity is in TRAIN_ARITIES.
              If HOLD_OUT_REST = True  → remaining arities go to val.
              If HOLD_OUT_REST = False → val is empty (just restrict training).

  "random" — seeded TRAIN_FRAC / (1-TRAIN_FRAC) split over all eligible tasks.

INIT_MODE controls how the fitted LbtAgent's α table is initialised:

  "uniform" — all α start at INIT_ALPHA (e.g. 1.0 = flat prior).
  "random"  — each α_k ~ U(INIT_ALPHA_LOW, INIT_ALPHA_HIGH).
"""

from pathlib import Path
import sys

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).parent / "results"

_RUN1_DIR = Path(__file__).parent.parent
import importlib.util as _ilu
_spec     = _ilu.spec_from_file_location("_run1_cfg", _RUN1_DIR / "config.py")
_run1_cfg = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_run1_cfg)

METADATA          = _run1_cfg.METADATA
BEHAVIOR_CSV_RUN0 = _run1_cfg.BEHAVIOR_CSV_RUN0
BEHAVIOR_CSV_RUN1 = _run1_cfg.BEHAVIOR_CSV_RUN1
BEH_ID_TO_TASK    = _run1_cfg.BEH_ID_TO_TASK

# ---------------------------------------------------------------------------
# Data handling
# ---------------------------------------------------------------------------
SEED               = _run1_cfg.SEED
USE_TRIAL_KINDS    = _run1_cfg.USE_TRIAL_KINDS
MIN_CATCH_PERF     = _run1_cfg.MIN_CATCH_PERF
MAIN_PERF_QUANTILE = _run1_cfg.MAIN_PERF_QUANTILE
MIN_TASK_ASSIGNMENTS = _run1_cfg.MIN_TASK_ASSIGNMENTS

# Minimum trials per (image, task) cell to include in training.
# Cells below this are dropped (main images often have very few trials).
MIN_TRIALS_PER_CELL = 2

# ---------------------------------------------------------------------------
# Decision rule
# ---------------------------------------------------------------------------
# NORMALIZED_UTILITY = True  — ΔU[k] = +1/|Z+| or -1/|Z-| (Bayes-corrected)
# NORMALIZED_UTILITY = False — ΔU[k] = +1 or -1  (original argmax rule)
NORMALIZED_UTILITY = True

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
N_EPOCHS  = 100
LR        = 0.1
GRAD_CLIP = 1.0
N_MC      = 5000       # MC samples for LbtAgent inference

# ---------------------------------------------------------------------------
# LbtAgent parameter initialisation
# ---------------------------------------------------------------------------
# INIT_MODE = "uniform" — all α start at INIT_ALPHA (e.g. 1.0 = flat prior)
# INIT_MODE = "random"  — each α_k ~ U(INIT_ALPHA_LOW, INIT_ALPHA_HIGH)
INIT_MODE       = "random"
INIT_ALPHA      = 1.0
INIT_ALPHA_LOW  = 0.6
INIT_ALPHA_HIGH = 0.7
INIT_SEED       = 0

# ---------------------------------------------------------------------------
# Task split
# ---------------------------------------------------------------------------
SPLIT_MODE    = "all"       # "all" | "arity" | "random"

# Used when SPLIT_MODE == "arity":
TRAIN_ARITIES = [4]     # arities included in training
HOLD_OUT_REST = True        # True → remaining arities go to val; False → no val

# Used when SPLIT_MODE == "random":
TRAIN_FRAC  = 0.80
SPLIT_SEED  = 0

_nu_tag = "norm" if NORMALIZED_UTILITY else "raw"
RUN_TAG  = f"lbt_noisy_{SPLIT_MODE}_{_nu_tag}"

# ---------------------------------------------------------------------------
# Task split — computed at import from eligible tasks in the real data
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
        return train, val

    elif SPLIT_MODE == "random":
        import numpy as np
        rng     = np.random.default_rng(SPLIT_SEED)
        arr     = np.array(all_eligible)
        idx     = rng.permutation(len(arr))
        n_train = int(round(len(arr) * TRAIN_FRAC))
        return sorted(arr[idx[:n_train]].tolist()), sorted(arr[idx[n_train:]].tolist())

    else:
        raise ValueError(
            f"Unknown SPLIT_MODE {SPLIT_MODE!r}. Choose 'all', 'arity', or 'random'."
        )


TRAIN_TASKS, VAL_TASKS = _compute_split()

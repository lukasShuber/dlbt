"""
Configuration for behavior run1 / 07_task_generalization.

Scientific question:
    Does DLBT generalize to task types not seen during training?

Design:
    Training budget = k tasks × all available trials per task, where
    k = number of eligible 1-arity tasks (the reference condition).

    Five training conditions (x-axis categories):
      1-arity   sample k tasks from 1-arity pool, all their trials
      2-arity   sample k tasks from 2-arity pool, all their trials
      3-arity   sample k tasks from 3-arity pool, all their trials
      4-arity   sample k tasks from 4-arity pool, all their trials
      random    sample k tasks uniformly from ALL tasks

    For each seed, a fresh sample of k tasks is drawn (resampling across
    seeds tests both weight and data variation).  1-arity has ≤ k tasks so
    every seed sees the same set → near-zero variance there.

    Probe evaluation: all probe images × tasks NOT used in training.
    This measures task generalization (transfer to unseen task types).

Reference lines (trained fresh on all tasks / all data, N_SEEDS seeds):
    Full DLBT   — evaluated on ALL probe images × ALL tasks
    Full SLDA   — evaluated on ALL probe images × ALL tasks
    Chance      — P=0.5 baseline

Run from repo root:
    python experiments/behavior/run1/07_task_generalization/run.py
    python experiments/behavior/run1/07_task_generalization/analysis.py
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
# Seeds
# ---------------------------------------------------------------------------
N_SEEDS = 1
SEEDS   = [42]

# ---------------------------------------------------------------------------
# Training (DLBT, frozen encoder)
# ---------------------------------------------------------------------------
N_EPOCHS           = 1000
PATIENCE           = 200
LR                 = 0.01
N_MC               = 1000
NORMALIZED_UTILITY = True
MAPPER_HIDDEN      = None

# ---------------------------------------------------------------------------
# Encoder freeze flags
# ---------------------------------------------------------------------------
FREEZE_ENCODER_DLBT = True   # False → Phase 2 attnpool fine-tuning for DLBT
FREEZE_ENCODER_SLDA = True   # False → Phase 2 attnpool fine-tuning for SLDA

# ---------------------------------------------------------------------------
# Training — Phase 2 (attnpool fine-tuning)
# ---------------------------------------------------------------------------
N_EPOCHS_PHASE2 = 3000
PATIENCE_PHASE2 = 50
LR_ATTNPOOL     = 1e-5

# ---------------------------------------------------------------------------
# DLBT base model  (symmetric Dirichlet α = BASE_CONCENTRATION)
# Under normalised utility SEU logit ≈ 0 → P(right) = 0.5.
# ---------------------------------------------------------------------------
BASE_CONCENTRATION = 1000.0

# ---------------------------------------------------------------------------
# Mapper initialisation
# ---------------------------------------------------------------------------
INIT_MODE       = "random"
INIT_ALPHA_LOW  = 0.6
INIT_ALPHA_HIGH = 0.7

# ---------------------------------------------------------------------------
# SLDA (reference line — all tasks, all data)
# ---------------------------------------------------------------------------
SLDA_Cs       = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
SLDA_MAX_ITER = 1000

# ---------------------------------------------------------------------------
# X-axis category order
# ---------------------------------------------------------------------------
ARITY_CONDITIONS = ["1-arity", "2-arity", "3-arity", "4-arity", "random"]

# ---------------------------------------------------------------------------
# Run tag  (encodes encoder settings so different runs don't overwrite each other)
# ---------------------------------------------------------------------------
_enc_dlbt = "frozen" if FREEZE_ENCODER_DLBT else "attnpool"
_enc_slda = "frozen" if FREEZE_ENCODER_SLDA else "attnpool"
RUN_TAG = f"task_generalization_dlbt_{_enc_dlbt}_slda_{_enc_slda}"

# ---------------------------------------------------------------------------
# Plot colours
# ---------------------------------------------------------------------------
C_DLBT      = "#C0392B"   # DLBT ablations — red
C_SLDA_REF  = "#7D3C98"   # SLDA full reference — purple
C_DLBT_REF  = "#C0392B"   # DLBT full reference — red  (matches 021)
C_CHANCE    = "#999999"   # random guesser — gray
C_SEED      = "#AAAAAA"   # per-seed scatter dots — light gray
ARITY_COLOR = {1: "#2a6fb5", 2: "#43AA8B", 3: "#E76F51", 4: "#9B5DE5",
               "random": "#F4A261"}


# ---------------------------------------------------------------------------
# Helper: eligible tasks grouped by arity
# ---------------------------------------------------------------------------
def eligible_tasks(df_filtered):
    tasks = _run1_cfg.eligible_tasks(df_filtered, MIN_TASK_ASSIGNMENTS)
    return sorted(tasks, key=lambda t: (t.count("_and_"), t))


def tasks_by_arity(all_tasks_ordered: list[str]) -> dict[int, list[str]]:
    """Return {arity: [task_name, ...]} for arity in 1..4."""
    out: dict[int, list[str]] = {}
    for t in all_tasks_ordered:
        a = t.count("_and_") + 1
        out.setdefault(a, []).append(t)
    return out

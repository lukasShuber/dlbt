"""
Configuration for behavior run1 / 06_slda_finetuning.

Compares two SLDA variants on a budget sweep:

  Frozen SLDA (Phase 1 only):
    1. Fit per-task ridge decoders on frozen CLIP features.
    2. Optimize temperature τ per task on same frozen features.

  Attnpool SLDA (three phases):
    1. Fit per-task ridge decoders on frozen CLIP features (no τ).
    2. Fine-tune CLIP attnpool through those fixed decoders (NLL, τ=1).
    3. Re-optimize τ per task using the fine-tuned features.

Temperature is always fit last — on whichever features are current at that
point.  In the frozen path this is trivially the frozen CLIP features; in the
attnpool path the features have been updated by Phase 2, so τ is meaningful.

Run from repo root:
    python experiments/behavior/run1/06_slda_finetuning/run.py
    python experiments/behavior/run1/06_slda_finetuning/analysis.py
"""

from pathlib import Path

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
# Budget grid  (same as 02)
# ---------------------------------------------------------------------------
TRIAL_BUDGETS: list[int] = sorted({
    int(round(10 ** (lo + k / 3)))
    for lo in range(2, 5)
    for k in range(3)
} | {100_000})

# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------
N_SEEDS = 5
SEEDS   = [42, 43, 44, 45, 46]

# ---------------------------------------------------------------------------
# Fast-pass mode
# ---------------------------------------------------------------------------
FAST_PASS = False

# ---------------------------------------------------------------------------
# Phase 2 — attnpool fine-tuning
# ---------------------------------------------------------------------------
N_EPOCHS_ATTNPOOL   = 3000
PATIENCE_ATTNPOOL   = 50
LR_ATTNPOOL         = 1e-5
BATCH_SIZE_ATTNPOOL = 128

# ---------------------------------------------------------------------------
# Run tag
# ---------------------------------------------------------------------------
RUN_TAG = "slda_finetuning"

# ---------------------------------------------------------------------------
# Plot options
# ---------------------------------------------------------------------------
LOG_Y = True

# ---------------------------------------------------------------------------
# Plot colours
# ---------------------------------------------------------------------------
C_FROZEN   = "#7D3C98"   # frozen SLDA  — saturated purple
C_ATTNPOOL = "#A569BD"   # attnpool SLDA — lighter purple
C_RNDINI   = "#999999"   # reference lines


# ---------------------------------------------------------------------------
# Helper: eligible tasks  (same filter as 02)
# ---------------------------------------------------------------------------
def eligible_tasks(df_filtered):
    tasks = _run1_cfg.eligible_tasks(df_filtered, MIN_TASK_ASSIGNMENTS)
    return sorted(tasks, key=lambda t: (t.count("_and_"), t))

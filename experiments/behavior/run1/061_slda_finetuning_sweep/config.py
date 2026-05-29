"""
Configuration for behavior run1 / 061_slda_finetuning_sweep.

Budget sweep (same grid as 021) comparing:
  - Frozen SLDA   — Phase 1 only (LogisticRegressionCV on frozen CLIP features)
  - Finetuned SLDA — Phase 1 + Phase 2 attnpool fine-tuning at a fixed lr

Two traces only; no DLBT, no anti-human condition.

Run from repo root:
    python experiments/behavior/run1/061_slda_finetuning_sweep/run.py
    python experiments/behavior/run1/061_slda_finetuning_sweep/analysis.py
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
# Budget grid  (trials per task) — mirrors 021_efficiency_main
# ---------------------------------------------------------------------------
TRIALS_PER_TASK: list[int] = sorted({
    int(round(10 ** (lo + k / 3)))
    for lo in range(1, 3)   # decades: 10^1, 10^2
    for k in range(3)
} | {1_000})
# → [10, 22, 46, 100, 215, 464, 1000]

# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------
N_SEEDS = 4
SEEDS   = [42, 43, 44, 45]

# ---------------------------------------------------------------------------
# Fast-pass mode  (quick smoke-test: smallest budget only)
# ---------------------------------------------------------------------------
FAST_PASS = False

# ---------------------------------------------------------------------------
# Phase 1 — LogisticRegressionCV  (matches 021_efficiency_main)
# ---------------------------------------------------------------------------
SLDA_Cs       = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
SLDA_MAX_ITER = 1000

# ---------------------------------------------------------------------------
# Phase 2 — attnpool fine-tuning (finetuned condition only)
# ---------------------------------------------------------------------------
LR_ATTNPOOL    = 1e-6
N_EPOCHS_PHASE2 = 3000
PATIENCE_PHASE2 = 50

# ---------------------------------------------------------------------------
# Run tag
# ---------------------------------------------------------------------------
RUN_TAG = f"slda_sweep_lr{LR_ATTNPOOL:.0e}"

# ---------------------------------------------------------------------------
# Plot colours
# ---------------------------------------------------------------------------
C_FROZEN   = "#7D3C98"   # frozen SLDA — purple (consistent with 021)
C_FINETUNE = "#1565C0"   # finetuned SLDA — dark blue
C_RNDINI   = "#999999"   # chance reference — gray

LOG_Y = True


# ---------------------------------------------------------------------------
# Helper: eligible tasks
# ---------------------------------------------------------------------------
def eligible_tasks(df_filtered):
    tasks = _run1_cfg.eligible_tasks(df_filtered, MIN_TASK_ASSIGNMENTS)
    return sorted(tasks, key=lambda t: (t.count("_and_"), t))

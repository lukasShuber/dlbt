"""
Configuration for behavior run1 / 06_slda_finetuning — SLDA Phase 2 sandbox.

Diagnostic experiment to understand SLDA Phase 2 (attnpool fine-tuning).
Trains on the full dataset (no budget sweep) and tracks training NLL and probe
cMSE across epochs to diagnose whether attnpool fine-tuning helps and which
learning rate works best.

Conditions:
  Phase 1 only          — fit LogReg on frozen CLIP features (reference)
  Phase 2 (lr=X)        — fine-tune attnpool through fixed Phase-1 decoders
  Phase 2 + refit scaler — same, but refit StandardScaler on fine-tuned features
                           after Phase 2 converges

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
# Seeds  (single seed — diagnostic run, not intended for statistical analysis)
# ---------------------------------------------------------------------------
N_SEEDS = 1
SEEDS   = [42]

# ---------------------------------------------------------------------------
# Phase 1 — LogisticRegressionCV  (matches 021_efficiency_main)
# ---------------------------------------------------------------------------
SLDA_Cs       = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
SLDA_MAX_ITER = 1000

# ---------------------------------------------------------------------------
# Phase 2 — attnpool fine-tuning
# ---------------------------------------------------------------------------
LR_ATTNPOOL_VARIANTS: list[float] = [1e-9, 1e-8, 1e-7]
N_EPOCHS_ATTNPOOL    = 3000
PATIENCE_ATTNPOOL    = 50
BATCH_SIZE_ATTNPOOL  = 128
EVAL_EVERY           = 25   # call probe cMSE hook every N epochs

# ---------------------------------------------------------------------------
# Run tag
# ---------------------------------------------------------------------------
RUN_TAG = "slda_sandbox"

# ---------------------------------------------------------------------------
# Plot colours
# ---------------------------------------------------------------------------
C_PHASE1  = "#7D3C98"   # Phase 1 only — purple (consistent with 021)
C_RNDINI  = "#999999"   # chance reference — gray

# Colors for Phase 2 LR variants: blue → teal gradient (3 shades)
C_PHASE2  = ["#1565C0", "#0097A7", "#2E7D32"]   # LR 1e-6, 1e-5, 1e-4

# Refit-scaler variant: same hues, slightly lighter / different marker
C_REFIT   = ["#5C9CE6", "#4DD0E1", "#81C784"]   # LR 1e-6, 1e-5, 1e-4


# ---------------------------------------------------------------------------
# Helper: eligible tasks
# ---------------------------------------------------------------------------
def eligible_tasks(df_filtered):
    tasks = _run1_cfg.eligible_tasks(df_filtered, MIN_TASK_ASSIGNMENTS)
    return sorted(tasks, key=lambda t: (t.count("_and_"), t))

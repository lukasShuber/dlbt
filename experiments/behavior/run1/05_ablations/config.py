"""
Configuration for behavior run1 / 05_ablations.

Budget sweep ablation comparing:
  - DLBT         : full model (MC Dirichlet integration, learned mapper)
  - DetBT        : deterministic beliefs (Dirichlet mean, learned mapper)
  - SLDA         : ridge decoder baseline (all tasks)
  - Oracle       : fixed beliefs from metadata latent state (no training)

Same data / sampling protocol as 023_efficiency_main:
  - Full task coverage only
  - Bootstrap fallback sampling
  - Dense log-spaced budget grid
  - Separate all-data point

Run from repo root:
    python experiments/behavior/run1/05_ablations/run.py
    python experiments/behavior/run1/05_ablations/analysis.py
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
# Budget grid
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
# Training
# ---------------------------------------------------------------------------
N_EPOCHS        = 1000
PATIENCE        = 200
N_EPOCHS_PHASE2 = 3000
PATIENCE_PHASE2 = 50
LR              = 0.01
LR_ATTNPOOL     = 1e-5
N_MC            = 1000
FREEZE_ENCODER      = False   
FREEZE_ENCODER_SLDA = True   
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

# ---------------------------------------------------------------------------
# Oracle agent hyperparameters
# ---------------------------------------------------------------------------
ORACLE_CONCENTRATION = 5.0   # Dirichlet mass on the true latent state bin
ORACLE_BACKGROUND    = 0.1   # Dirichlet mass on all other bins

_enc_dlbt = "frozen" if FREEZE_ENCODER      else "attnpool"
_enc_slda = "frozen" if FREEZE_ENCODER_SLDA else "attnpool"
RUN_TAG   = f"ablations_dlbt_{_enc_dlbt}_slda_{_enc_slda}_s{len(SEEDS)}"

# ---------------------------------------------------------------------------
# Plot options
# ---------------------------------------------------------------------------
LOG_Y = True

# ---------------------------------------------------------------------------
# Plot colours
# ---------------------------------------------------------------------------
C_DLBT    = "#C0392B"   # DLBT — red
C_DETBT   = "#2a6fb5"   # DetBT — blue
C_SLDA    = "#7D3C98"   # SLDA — purple
C_ORACLE  = "#E67E22"   # Oracle — orange
C_RANDONT = "#27AE60"   # RandOnt — green
C_RNDINI  = "#999999"   # reference lines


# ---------------------------------------------------------------------------
# Helper: eligible tasks
# ---------------------------------------------------------------------------
def eligible_tasks(df_filtered):
    tasks = _run1_cfg.eligible_tasks(df_filtered, MIN_TASK_ASSIGNMENTS)
    return sorted(tasks, key=lambda t: (t.count("_and_"), t))

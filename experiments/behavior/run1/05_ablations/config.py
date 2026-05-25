"""
Configuration for behavior run1 / 05_ablations.

Belief-representation ablation comparing:
  - DLBT            : full model (MC Dirichlet integration, learned mapper)
  - DetBT           : perceptual stochasticity ablation
                      (Dirichlet mean at train AND eval; no MC sampling)
  - OneHotBT        : perceptual uncertainty ablation
                      (train: mean like DetBT; eval: argmax one-hot, certain belief)
  - BehavSuperv     : no behavioral supervision reference — P=1 if true latent ∈ Z+,
                      P=0 otherwise; no learning, purely ground-truth-based

Same data / sampling protocol as 021_efficiency_main:
  - Full task coverage only
  - Bootstrap fallback sampling (trial-level)
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
# Budget grid  (trials per task — mirrors 021_efficiency_main)
# ---------------------------------------------------------------------------
TRIALS_PER_TASK: list[int] = sorted({
    int(round(10 ** (lo + k / 3)))
    for lo in range(1, 3)
    for k in range(3)
} | {1_000})
# → [10, 22, 46, 100, 215, 464, 1000]

# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------
N_SEEDS = 1
SEEDS   = [42]

# ---------------------------------------------------------------------------
# Fast-pass mode
# ---------------------------------------------------------------------------
FAST_PASS = False

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
N_EPOCHS           = 1000
PATIENCE           = 200
N_EPOCHS_PHASE2    = 3000
PATIENCE_PHASE2    = 50
LR                 = 0.01
LR_ATTNPOOL        = 1e-5
N_MC               = 1000
FREEZE_ENCODER     = False   # False → Phase 2 attnpool fine-tuning for DLBT
MAPPER_HIDDEN      = None
NORMALIZED_UTILITY = True

# ---------------------------------------------------------------------------
# DLBT base model  (symmetric Dirichlet α = BASE_CONCENTRATION → P ≈ 0.5)
# ---------------------------------------------------------------------------
BASE_CONCENTRATION = 1000.0

# ---------------------------------------------------------------------------
# Mapper initialisation
# ---------------------------------------------------------------------------
INIT_MODE       = "random"
INIT_ALPHA_LOW  = 0.6
INIT_ALPHA_HIGH = 0.7

_enc_dlbt = "frozen" if FREEZE_ENCODER else "attnpool"
RUN_TAG   = f"ablations_dlbt_{_enc_dlbt}_s{N_SEEDS}"

# ---------------------------------------------------------------------------
# Plot options
# ---------------------------------------------------------------------------
LOG_Y = True

# ---------------------------------------------------------------------------
# Plot colours
# ---------------------------------------------------------------------------
C_DLBT        = "#C0392B"   # DLBT — strong red
C_DETBT       = "#C95C48"   # DetBT (perc. stochasticity) — warm red-salmon
C_ONEHOT      = "#D4876A"   # OneHotBT (perc. uncertainty) — medium salmon
C_BEHAV_SUPER = "#8E44AD"   # No beh. supervision (ground-truth P=0/1) — purple
C_RNDINI      = "#999999"   # reference lines — gray


# ---------------------------------------------------------------------------
# Helper: eligible tasks
# ---------------------------------------------------------------------------
def eligible_tasks(df_filtered):
    tasks = _run1_cfg.eligible_tasks(df_filtered, MIN_TASK_ASSIGNMENTS)
    return sorted(tasks, key=lambda t: (t.count("_and_"), t))

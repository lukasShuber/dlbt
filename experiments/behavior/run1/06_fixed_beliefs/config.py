"""
Configuration for behavior run1 / 06_fixed_beliefs — oracle belief ablation.

OracleBTAgent uses fixed Dirichlet beliefs peaked at the true latent state
(from image metadata), bypassing the learned mapper entirely.  No behavioral
training is required.

The key configurable parameter is CONCENTRATION: how strongly the Dirichlet
peaks at the true latent state bin.  Moderate values (2–10) model genuine
perceptual uncertainty around the binarisation threshold.  The one-hot limit
(c → ∞) is already covered by 05_determ_beliefs.

Run from repo root:
    python experiments/behavior/run1/06_fixed_beliefs/run.py
    python experiments/behavior/run1/06_fixed_beliefs/analysis.py
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

BEHAVIOR_CSV_RUN0    = _run1_cfg.BEHAVIOR_CSV_RUN0
BEHAVIOR_CSV_RUN1    = _run1_cfg.BEHAVIOR_CSV_RUN1
BEH_ID_TO_TASK       = _run1_cfg.BEH_ID_TO_TASK

# ---------------------------------------------------------------------------
# Data handling
# ---------------------------------------------------------------------------
SEED                 = _run1_cfg.SEED
USE_TRIAL_KINDS      = _run1_cfg.USE_TRIAL_KINDS
MIN_CATCH_PERF       = _run1_cfg.MIN_CATCH_PERF
MAIN_PERF_QUANTILE   = _run1_cfg.MAIN_PERF_QUANTILE
MIN_TASK_ASSIGNMENTS = _run1_cfg.MIN_TASK_ASSIGNMENTS

# ---------------------------------------------------------------------------
# Oracle parameters
# ---------------------------------------------------------------------------
# Dirichlet mass at the true latent state dimension.
# Moderate values (2–10) model perceptual uncertainty around the threshold.
# Change this to explore the sensitivity of the oracle to concentration.
CONCENTRATION = 5.0

# Background mass on all other K-1 dimensions.
BACKGROUND = 0.1

# Normalise ΔU by arity (mirrors 02 and 022 settings).
NORMALIZED_UTILITY = True

RUN_TAG = "oracle_beliefs"

# ---------------------------------------------------------------------------
# Plot colours
# ---------------------------------------------------------------------------
C_ORACLE = "#C0392B"   # saturated red — distinct from purple SLDA, blue DLBT
C_SLDA   = "#7D3C98"

import matplotlib.pyplot as _plt
_BLUES = _plt.get_cmap("Blues")
_CMAP_OFFSETS = {0.10: 0.30, 0.25: 0.44, 0.50: 0.58, 0.75: 0.72, 1.00: 0.88}
def cov_color(frac: float):
    return _BLUES(_CMAP_OFFSETS.get(frac, 0.6))

ARITY_COLOR = {1: "#2a6fb5", 2: "#43AA8B", 3: "#E76F51", 4: "#9B5DE5"}


# ---------------------------------------------------------------------------
# Helper: eligible tasks
# ---------------------------------------------------------------------------
def eligible_tasks(df_filtered):
    tasks = _run1_cfg.eligible_tasks(df_filtered, MIN_TASK_ASSIGNMENTS)
    return sorted(tasks, key=lambda t: (t.count("_and_"), t))

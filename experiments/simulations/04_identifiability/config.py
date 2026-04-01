"""
Configuration for simulation 04 — identifiability.

Two oracle feature modes are compared:
  onehot  — K=16 one-hot state vector: strict identifiability test.
             Mapper learns one alpha vector per discrete latent state purely
             from behavioral observations.
  latent  — 4D continuous GT latent coords with a linear mapper:
             expressivity test. How well can a linear function approximate
             the nonlinear GT alpha function?
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
METADATA    = "stimuli/imgs_pink/metadata.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"

# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------
N_TRIALS           = 5000    # per (image, task) pair — large for clean signal
PEAK               = 15.0
BASE_CONCENTRATION = 1.0
BETA               = 5.0
SCALE_BETA         = 10.0

# ---------------------------------------------------------------------------
# Training  (single run, no seeds — deterministic oracle features)
# ---------------------------------------------------------------------------
N_EPOCHS  = 5000
PATIENCE  = 100
LR        = 5e-2
N_MC      = 100    # small mapper converges easily; fewer MC samples needed

# ---------------------------------------------------------------------------
# Feature modes
# ---------------------------------------------------------------------------
FEATURE_MODES = ["onehot", "latent"]
FEATURE_DIMS  = {"onehot": 16, "latent": 4}
MODE_LABELS   = {
    "onehot": "One-hot state  (K=16)",
    "latent": "GT latent coords  (4D linear)",
}
MODE_COLORS = {
    "onehot": "#E76F51",   # coral
    "latent": "#457B9D",   # steel blue
}

# ---------------------------------------------------------------------------
# Tasks — ALL tasks used for fitting (no held-out)
# ---------------------------------------------------------------------------
ALL_TASKS = [
    # simple
    "left_right", "transparent", "glossy", "large",
    "left", "opaque", "matte", "small",
    # 2-way: lr × material
    "right_and_transparent", "left_and_transparent",
    "right_and_glossy",      "left_and_glossy",
    # 2-way: material × material
    "transparent_and_glossy",
    # 2-way: sl × material
    "large_and_transparent", "large_and_glossy",
    # 3-way
    "right_and_transparent_and_glossy",
    "left_and_transparent_and_glossy",
    "large_and_transparent_and_glossy",
    # spatial × spatial conjunctions (held-out in other experiments)
    "right_and_large",
    "left_and_large",
    "right_and_large_and_glossy",
    "right_and_large_and_transparent",
]

"""
Configuration for simulation 04 — LBT recovery.

SPLIT_MODE controls which tasks are used for training vs evaluation:

  "all"    — train on all 80 tasks, no val.

  "arity"  — train on tasks whose arity is in TRAIN_ARITIES.
              If HOLD_OUT_REST = True  → remaining arities go to val.
              If HOLD_OUT_REST = False → val is empty (just restrict training).
              e.g. TRAIN_ARITIES = [1], HOLD_OUT_REST = True  → train 1-way, eval 2/3/4-way
                   TRAIN_ARITIES = [1], HOLD_OUT_REST = False → train 1-way only, no val

  "random" — random TRAIN_FRAC / (1-TRAIN_FRAC) split over all 80 tasks,
              seeded by SPLIT_SEED for reproducibility.
"""

from pathlib import Path
from itertools import combinations, product

from dlbt.data.task import get_task

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
METADATA    = "stimuli/imgs/metadata.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"

# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
SEED          = 42
N_TRIALS      = 100       # synthetic trials per (image, task) cell
N_MC          = 50000      # MC samples for LbtAgent inference
N_EPOCHS      = 200
LR            = 0.95
GRAD_CLIP     = 1.0

# ---------------------------------------------------------------------------
# Ground-truth observer
# ---------------------------------------------------------------------------
# GT_MODE = "peaked"     — α[true_state] = CONCENTRATION, α[other] = CONCENTRATION_BG
# GT_MODE = "graded"    — α_k depends on # shared features with true_state (0–4):
#                           0 shared → GRADED_LEVELS[0]  (minimum)
#                           ...
#                           4 shared → GRADED_LEVELS[4]  (= true state, peak)
# GT_MODE = "factorized" — uses real image continuous metadata (x, transparency,
#                           glossiness, scale); per-feature probabilities via sigmoid,
#                           α_k = λ × ∏ p(feature_k), λ = BASE_CONC + PEAK × clarity
# GT_MODE = "random"    — α_k ~ U(GT_ALPHA_LOW, GT_ALPHA_HIGH) for all k, i.i.d.
GT_MODE          = "factorized"
CONCENTRATION    = 5.0     # used when GT_MODE == "peaked": peak value
CONCENTRATION_BG = 0.1     # used when GT_MODE == "peaked": background value
GRADED_LEVELS    = [0.01, 2.0, 4.0, 8.0, 100.0]  # used when GT_MODE == "graded"
                                                   # index = number of shared features
# used when GT_MODE == "factorized":
BETA             = 8.0     # sigmoid slope for left/right, transparency, glossiness
SCALE_BETA       = 15.0    # sigmoid slope for scale
BASE_CONCENTRATION = 0.1   # baseline added to λ before scaling
PEAK             = 30.0    # added concentration for maximally clear images
MIN_LAM          = 3.0     # floor on λ — ensures signal even for ambiguous images
SHARPNESS        = 2.0     # q = q**SHARPNESS before normalising — sharpens the peak
GT_ALPHA_LOW     = 1.0     # used when GT_MODE == "random"
GT_ALPHA_HIGH    = 2.0     # used when GT_MODE == "random"
GT_SEED          = 1       # used when GT_MODE == "random"

# ---------------------------------------------------------------------------
# LbtAgent parameter initialisation (NOT the ground-truth observer)
# Controls how the fitted agent's α table is initialised before training.
# The ground-truth is set by GT_MODE / CONCENTRATION (peaked at true_state, 1.0 elsewhere).
# ---------------------------------------------------------------------------
# INIT_MODE = "uniform" — all α start at INIT_ALPHA (set to any level, e.g. 1.0, 10.0)
# INIT_MODE = "random"  — each α_k drawn independently from
#                         Uniform(INIT_ALPHA_LOW, INIT_ALPHA_HIGH)
INIT_MODE       = "random"
INIT_ALPHA      = 1.0          # used when INIT_MODE == "uniform"
INIT_ALPHA_LOW  = 0.1          # used when INIT_MODE == "random"
INIT_ALPHA_HIGH = 1.0          # used when INIT_MODE == "random"
INIT_SEED       = 0            # used when INIT_MODE == "random"

# ---------------------------------------------------------------------------
# Task split
# ---------------------------------------------------------------------------
SPLIT_MODE    = "all"    # "all" | "arity" | "random"

# Used when SPLIT_MODE == "arity":
TRAIN_ARITIES = [2,3]       # arities included in training, e.g. [1], [1, 2]
HOLD_OUT_REST = True      # True → remaining arities go to val; False → no val

# Used when SPLIT_MODE == "random":
TRAIN_FRAC   = 0.80
SPLIT_SEED   = 0

# ---------------------------------------------------------------------------
# Task generation
# ---------------------------------------------------------------------------

def all_tasks():
    """Return all 80 tasks: 8 one-way + 24 two-way + 32 three-way + 16 four-way."""
    features = [
        ("right",       "left"),
        ("transparent", "opaque"),
        ("glossy",      "matte"),
        ("large",       "small"),
    ]
    tasks = []
    for n_way in range(1, 5):
        for dims in combinations(range(4), n_way):
            for values in product(*[features[d] for d in dims]):
                tasks.append(get_task("_and_".join(values)))
    return tasks


def compute_split():
    """
    Return (train_tasks, val_tasks) according to SPLIT_MODE.
    val_tasks is empty when SPLIT_MODE == "all".
    """
    tasks = all_tasks()

    if SPLIT_MODE == "all":
        return tasks, []

    elif SPLIT_MODE == "arity":
        train = [t for t in tasks if t.name.count("_and_") + 1 in TRAIN_ARITIES]
        val   = ([t for t in tasks if t.name.count("_and_") + 1 not in TRAIN_ARITIES]
                 if HOLD_OUT_REST else [])
        return train, val

    elif SPLIT_MODE == "random":
        import numpy as np
        rng     = np.random.default_rng(SPLIT_SEED)
        idx     = rng.permutation(len(tasks))
        n_train = int(round(len(tasks) * TRAIN_FRAC))
        train   = [tasks[i] for i in idx[:n_train]]
        val     = [tasks[i] for i in idx[n_train:]]
        return train, val

    else:
        raise ValueError(
            f"Unknown SPLIT_MODE {SPLIT_MODE!r}. Choose 'all', 'arity', or 'random'."
        )


TRAIN_TASKS, VAL_TASKS = compute_split()

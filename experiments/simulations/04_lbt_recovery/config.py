"""
Configuration for simulation 04 — LBT recovery.

SPLIT_MODE controls which tasks are used for training vs evaluation:

  "all"    — train on all 80 tasks (no held-out split).

  "arity"  — train on tasks whose arity is in TRAIN_ARITIES; hold out the rest.
              e.g. TRAIN_ARITIES = [1]       → train 1-way, eval 2/3/4-way
                   TRAIN_ARITIES = [1, 2]    → train 1+2-way, eval 3/4-way
                   TRAIN_ARITIES = [1, 2, 3] → train 1+2+3-way, eval 4-way

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
N_MC          = 1000      # MC samples for LbtAgent inference
N_EPOCHS      = 300
LR            = 1e-2
GRAD_CLIP     = 1.0

# Concentration sweep: α at the true latent state; all others = 1.0.
CONCENTRATIONS = [5.0]    # extend e.g. to [2, 5, 10, 50] for a sweep

# ---------------------------------------------------------------------------
# LbtAgent parameter initialisation (NOT the ground-truth observer)
# Controls how the fitted agent's α table is initialised before training.
# The ground-truth is set by CONCENTRATIONS (peaked at true_state, 1.0 elsewhere).
# ---------------------------------------------------------------------------
# INIT_MODE = "uniform" — all α start at INIT_ALPHA (set to any level, e.g. 1.0, 10.0)
# INIT_MODE = "random"  — each α_k drawn independently from
#                         Uniform(INIT_ALPHA_LOW, INIT_ALPHA_HIGH)
INIT_MODE       = "uniform"
INIT_ALPHA      = 1.0          # used when INIT_MODE == "uniform"
INIT_ALPHA_LOW  = 0.5          # used when INIT_MODE == "random"
INIT_ALPHA_HIGH = 3.0          # used when INIT_MODE == "random"
INIT_SEED       = 0            # used when INIT_MODE == "random"

# ---------------------------------------------------------------------------
# Task split
# ---------------------------------------------------------------------------
SPLIT_MODE   = "all"    # "all" | "arity" | "random"

# Used when SPLIT_MODE == "arity":
# List of arities (1–4) to include in the training set.
TRAIN_ARITIES = [1]       # e.g. [1], [1, 2], [1, 2, 3]

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
        val   = [t for t in tasks if t.name.count("_and_") + 1 not in TRAIN_ARITIES]
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

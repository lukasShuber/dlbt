"""
Simulation 04 — Concentration diagnostic.

For a peaked GT observer (alpha[true_state] = c, alpha[other] = 1.0),
sweep over a range of concentration values c and compute the oracle P(right)
for every (image × task) cell — no fitting, just the generative model.

Goal: visualise how the sharpness of the GT Dirichlet determines how
deterministic the simulated choices are, across all task arities.

Output:
  plot_concentration_{tag}.png  — strip plot: x = concentration, y = P(right),
                                   colour = arity

Run from repo root:
    python experiments/simulations/04_lbt_recovery/run_concentration.py
"""

import sys
from collections import defaultdict
from itertools import combinations, product
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[3]))

from dlbt.constants import K, DIM_LEFT_RIGHT, DIM_TRANSP, DIM_GLOSS, DIM_SMALL_LARGE
from dlbt.data.image_ref import load_image_refs, image_refs_as_list
from dlbt.data.task import get_task

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

# ---------------------------------------------------------------------------
# Settings (edit here)
# ---------------------------------------------------------------------------
CONCENTRATIONS  = [1, 2, 5, 10, 20, 50]   # peak α values to sweep
CONCENTRATION_BG = 1.0                     # background α (non-true states)
N_MC            = 10_000                   # MC samples for oracle P(right)
SEED            = 42
RESULTS_DIR     = cfg.RESULTS_DIR
METADATA        = cfg.METADATA

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
ARITY_COLOR = {1: "#2a6fb5", 2: "#43AA8B", 3: "#E76F51", 4: "#9B5DE5"}

def _arity(task) -> int:
    return task.name.count("_and_") + 1


def all_tasks():
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


def oracle_p_right(alpha: np.ndarray, task, n_mc: int, rng) -> float:
    b = rng.dirichlet(alpha, size=n_mc)
    return float((b @ task.delta_u > 0).mean())


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
plots_dir = RESULTS_DIR / "plots"
plots_dir.mkdir(exist_ok=True)

rng        = np.random.default_rng(SEED)
tasks      = all_tasks()

# Load image refs — one representative per latent state
print("Loading image refs...")
refs_dict = load_image_refs(METADATA)
refs_all  = image_refs_as_list(refs_dict)
by_state  = defaultdict(list)
for r in refs_all:
    by_state[r.latent_state].append(r)
probe_refs = [by_state[k][0] for k in range(K) if by_state[k]]
assert len(probe_refs) == K

print(f"  Probe images: {len(probe_refs)}  Tasks: {len(tasks)}")

# ---------------------------------------------------------------------------
# Sweep concentrations — compute oracle P(right) for each cell
# ---------------------------------------------------------------------------
rows = []   # list of dicts

for c in CONCENTRATIONS:
    print(f"Concentration c={c} ...")
    for ref in probe_refs:
        alpha = np.full(K, CONCENTRATION_BG)
        alpha[ref.latent_state] = float(c)

        for task in tasks:
            p = oracle_p_right(alpha, task, n_mc=N_MC, rng=rng)
            rows.append({
                "concentration": c,
                "arity":         _arity(task),
                "p_right":       p,
            })

print(f"Total cells computed: {len(rows)}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
import pandas as pd

df = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(7, 4))

arities = sorted(df["arity"].unique())
jitter_width = 0.15

for arity in arities:
    sub = df[df["arity"] == arity]
    color = ARITY_COLOR[arity]

    for c in CONCENTRATIONS:
        vals = sub[sub["concentration"] == c]["p_right"].values
        x    = np.full(len(vals), float(c))
        x   += rng.uniform(-jitter_width, jitter_width, size=len(vals))
        ax.scatter(
            x, vals,
            color  = color,
            alpha  = 0.25,
            s      = 8,
            lw     = 0,
            label  = f"{arity}-way" if c == CONCENTRATIONS[0] else None,
        )

ax.set_xscale("log")
ax.set_xticks(CONCENTRATIONS)
ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
ax.set_xlabel("Concentration  (peak α)", fontsize=11)
ax.set_ylabel("Oracle  P(right)", fontsize=11)
ax.set_title("Oracle P(right) vs. GT concentration", fontsize=12)
ax.axhline(0.5, color="gray", lw=0.8, ls="--")
ax.set_ylim(-0.02, 1.02)
ax.legend(title="arity", fontsize=9, title_fontsize=9,
          loc="center right", framealpha=0.8)

plt.tight_layout()
out_path = plots_dir / "plot_concentration.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")

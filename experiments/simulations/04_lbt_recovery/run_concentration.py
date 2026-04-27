# """
# Simulation 04 — Concentration diagnostic.

# For a peaked GT observer (alpha[true_state] = c, alpha[other] = 1.0),
# sweep over a range of concentration values c and compute the oracle P(right)
# for every (image × task) cell — no fitting, just the generative model.

# Goal: visualise how the sharpness of the GT Dirichlet determines how
# deterministic the simulated choices are, across all task arities.

# Output:
#   plot_concentration_{tag}.png  — strip plot: x = concentration, y = P(right),
#                                    colour = arity

# Run from repo root:
#     python experiments/simulations/04_lbt_recovery/run_concentration.py
# """

# import sys
# from collections import defaultdict
# from itertools import combinations, product
# from pathlib import Path

# import matplotlib.pyplot as plt
# import matplotlib.ticker as mticker
# import numpy as np

# sys.path.insert(0, str(Path(__file__).parents[3]))

# from dlbt.constants import K, DIM_LEFT_RIGHT, DIM_TRANSP, DIM_GLOSS, DIM_SMALL_LARGE
# from dlbt.data.image_ref import load_image_refs, image_refs_as_list
# from dlbt.data.task import get_task

# sys.path.insert(0, str(Path(__file__).parent))
# import config as cfg

# # ---------------------------------------------------------------------------
# # Settings (edit here)
# # ---------------------------------------------------------------------------
# CONCENTRATIONS  = [1, 2, 5, 10, 20, 50, 1000, np.inf]  # np.inf = one-hot (deterministic)
# CONCENTRATION_BG = 1.0                     # background α (non-true states)
# N_MC            = 10_000                   # MC samples for oracle P(right)
# SEED            = 42
# RESULTS_DIR     = cfg.RESULTS_DIR
# METADATA        = cfg.METADATA

# # ---------------------------------------------------------------------------
# # Helpers
# # ---------------------------------------------------------------------------
# ARITY_COLOR = {1: "#2a6fb5", 2: "#43AA8B", 3: "#E76F51", 4: "#9B5DE5"}

# def _arity(task) -> int:
#     return task.name.count("_and_") + 1


# def all_tasks():
#     features = [
#         ("right",       "left"),
#         ("transparent", "opaque"),
#         ("glossy",      "matte"),
#         ("large",       "small"),
#     ]
#     tasks = []
#     for n_way in range(1, 5):
#         for dims in combinations(range(4), n_way):
#             for values in product(*[features[d] for d in dims]):
#                 tasks.append(get_task("_and_".join(values)))
#     return tasks


# def oracle_p_right(alpha: np.ndarray, task, n_mc: int, rng,
#                    true_state: int = None) -> float:
#     """MC estimate of P(right). If true_state is given and alpha is one-hot,
#     return the deterministic result (delta_u[true_state] > 0)."""
#     if true_state is not None and np.isinf(alpha[true_state]):
#         return float(task.delta_u[true_state] > 0)
#     b = rng.dirichlet(alpha, size=n_mc)
#     return float((b @ task.delta_u > 0).mean())


# # ---------------------------------------------------------------------------
# # Setup
# # ---------------------------------------------------------------------------
# RESULTS_DIR.mkdir(parents=True, exist_ok=True)
# plots_dir = RESULTS_DIR / "plots"
# plots_dir.mkdir(exist_ok=True)

# rng        = np.random.default_rng(SEED)
# tasks      = all_tasks()

# # Load image refs — one representative per latent state
# print("Loading image refs...")
# refs_dict = load_image_refs(METADATA)
# refs_all  = image_refs_as_list(refs_dict)
# by_state  = defaultdict(list)
# for r in refs_all:
#     by_state[r.latent_state].append(r)
# probe_refs = [by_state[k][0] for k in range(K) if by_state[k]]
# assert len(probe_refs) == K

# print(f"  Probe images: {len(probe_refs)}  Tasks: {len(tasks)}")

# # ---------------------------------------------------------------------------
# # Sweep concentrations — compute oracle P(right) for each cell
# # ---------------------------------------------------------------------------
# rows = []   # list of dicts

# for c in CONCENTRATIONS:
#     label = "∞" if np.isinf(c) else str(c)
#     print(f"Concentration c={label} ...")
#     for ref in probe_refs:
#         alpha = np.full(K, CONCENTRATION_BG)
#         alpha[ref.latent_state] = c          # np.inf handled in oracle_p_right

#         for task in tasks:
#             p = oracle_p_right(alpha, task, n_mc=N_MC, rng=rng,
#                                true_state=ref.latent_state)
#             rows.append({
#                 "concentration": c,
#                 "arity":         _arity(task),
#                 "p_right":       p,
#             })

# print(f"Total cells computed: {len(rows)}")

# # ---------------------------------------------------------------------------
# # Plot
# # ---------------------------------------------------------------------------
# import pandas as pd

# df = pd.DataFrame(rows)

# # For plotting: replace np.inf with a sentinel x-value just beyond max finite conc.
# finite_concs = [c for c in CONCENTRATIONS if not np.isinf(c)]
# ONEHOT_X     = finite_concs[-1] * 3          # e.g. 3000 if last finite is 1000
# x_positions  = {c: (ONEHOT_X if np.isinf(c) else c) for c in CONCENTRATIONS}
# x_ticks      = [x_positions[c] for c in CONCENTRATIONS]
# x_labels     = ["∞" if np.isinf(c) else str(c) for c in CONCENTRATIONS]

# fig, ax = plt.subplots(figsize=(8, 4.5))

# arities = sorted(df["arity"].unique())
# jitter_width = 0.15

# for arity in arities:
#     sub = df[df["arity"] == arity]
#     color = ARITY_COLOR[arity]

#     for c in CONCENTRATIONS:
#         vals = sub[sub["concentration"] == c]["p_right"].values
#         xc   = x_positions[c]
#         x    = np.full(len(vals), float(xc))
#         x   *= np.exp(rng.uniform(-jitter_width, jitter_width, size=len(vals)))
#         ax.scatter(
#             x, vals,
#             color  = color,
#             alpha  = 0.40,
#             s      = 18,
#             lw     = 0,
#             label  = f"{arity}-way" if c == CONCENTRATIONS[0] else None,
#         )

# ax.set_xscale("log")
# ax.set_xticks(x_ticks)
# ax.set_xticklabels(x_labels)
# ax.tick_params(axis="both", labelsize=11)
# ax.set_xlabel("Concentration  (peak α)", fontsize=13)
# ax.set_ylabel("Oracle  P(right)", fontsize=13)
# ax.set_title("Oracle P(right) vs. GT concentration", fontsize=14)
# ax.axhline(0.5, color="gray", lw=1.2, ls="--")
# ax.set_ylim(-0.02, 1.02)
# ax.legend(title="arity", fontsize=11, title_fontsize=11,
#           loc="center right", framealpha=0.9)

# plt.tight_layout()
# out_path = plots_dir / "plot_concentration.png"
# fig.savefig(out_path, dpi=150, bbox_inches="tight")
# plt.close(fig)
# print(f"Saved: {out_path}")

# # ---------------------------------------------------------------------------
# # Plot 2 — grid of histograms: rows = arity, cols = concentration
# # ---------------------------------------------------------------------------
# n_rows = len(arities)
# n_cols = len(CONCENTRATIONS)

# fig2, axes = plt.subplots(
#     n_rows, n_cols,
#     figsize=(2.6 * n_cols, 2.4 * n_rows),
#     sharex=True, sharey=False,
# )

# for row, arity in enumerate(arities):
#     for col, c in enumerate(CONCENTRATIONS):
#         ax = axes[row, col]
#         vals = df[(df["arity"] == arity) & (df["concentration"] == c)]["p_right"].values
#         ax.hist(vals, bins=20, range=(0, 1),
#                 color=ARITY_COLOR[arity], alpha=0.85, edgecolor="none")
#         ax.axvline(0.5, color="gray", lw=1.2, ls="--")
#         ax.set_xlim(0, 1)
#         ax.tick_params(labelsize=9)
#         if col > 0:
#             ax.tick_params(labelleft=False)

#         if row == 0:
#             clabel = "∞" if np.isinf(c) else str(c)
#             ax.set_title(f"c = {clabel}", fontsize=11, fontweight="bold")
#         if col == 0:
#             ax.set_ylabel(f"{arity}-way", fontsize=11,
#                           color=ARITY_COLOR[arity], fontweight="bold")

# fig2.supxlabel("Oracle P(right)", fontsize=12, y=0.01)
# fig2.suptitle("P(right) distributions  [arity × concentration]", fontsize=14, y=1.01)
# plt.tight_layout()
# out_path2 = plots_dir / "plot_concentration_histograms.png"
# fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
# plt.close(fig2)
# print(f"Saved: {out_path2}")


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# -----------------------------
# Config
# -----------------------------
K = 16
DIMS = 4
BASE = 1.0
N_MC = 20_000
SEED = 0

concentrations = np.logspace(-1, 3, 45)
arities = [1, 2, 3, 4]
rng = np.random.default_rng(SEED)

colors = {

    1: "#2a6fb5",

    2: "#43AA8B",

    3: "#E76F51",

    4: "#9B5DE5",

}

# -----------------------------
# Task + belief helpers
# -----------------------------
def make_and_task(active_dims):
    """Delta U = +1 iff all active dimensions are 1, else -1."""
    delta_u = np.empty(K, dtype=float)

    for k in range(K):
        bits = [(k >> d) & 1 for d in range(DIMS)]
        is_positive = all(bits[d] == 1 for d in active_dims)
        delta_u[k] = 1.0 if is_positive else -1.0

    return delta_u


def dirichlet_alpha(true_k, peak_concentration, base=BASE):
    """Dirichlet with base concentration everywhere and peak added to true state."""
    alpha = np.full(K, base, dtype=float)
    alpha[true_k] += peak_concentration
    return alpha


def p_right_for_state(true_k, delta_u, peak_concentration, n_mc=N_MC):
    """Estimate P(right) for a true latent state under Dirichlet-SEU."""
    alpha = dirichlet_alpha(true_k, peak_concentration)
    beliefs = rng.dirichlet(alpha, size=n_mc)
    return np.mean((beliefs @ delta_u) > 0)


# -----------------------------
# Run simulation
# -----------------------------
results = {}

for arity in arities:
    delta_u = make_and_task(active_dims=list(range(arity)))

    positive_states = np.where(delta_u == 1)[0]
    negative_states = np.where(delta_u == -1)[0]

    mean_pos = []
    mean_neg = []

    for c in concentrations:
        p_pos = np.mean([
            p_right_for_state(k, delta_u, c)
            for k in positive_states
        ])

        p_neg = np.mean([
            p_right_for_state(k, delta_u, c)
            for k in negative_states
        ])

        mean_pos.append(p_pos)
        mean_neg.append(p_neg)

    results[arity] = {
        "positive": np.array(mean_pos),
        "negative": np.array(mean_neg),
        "n_positive": len(positive_states),
        "n_negative": len(negative_states),
    }


# -----------------------------
# Plot
# -----------------------------
plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 18,
    "axes.labelsize": 17,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "lines.linewidth": 3.0,
})

fig, ax = plt.subplots(figsize=(10, 6.5))

for arity in arities:
    color = colors[arity]
    mean_pos = results[arity]["positive"]
    mean_neg = results[arity]["negative"]

    ax.plot(
        concentrations,
        mean_pos,
        color=color,
        linestyle="-",
        linewidth=3.4,
    )

    ax.plot(
        concentrations,
        mean_neg,
        color=color,
        linestyle="--",
        linewidth=3.4,
    )

ax.axhline(
    0.5,
    color="black",
    linestyle=":",
    linewidth=2.2,
    alpha=0.8,
)

ax.set_xscale("log")
ax.set_xlim(concentrations.min(), concentrations.max())
ax.set_ylim(-0.03, 1.03)

ax.set_xlabel("Peak concentration on true latent state")
ax.set_ylabel("Predicted $P(\\mathrm{right})$")
ax.set_title("Only near one-hot beliefs remove the arity-induced bias")

ax.grid(True, which="major", alpha=0.25)
ax.grid(True, which="minor", alpha=0.10)

# -----------------------------
# Custom legend
# -----------------------------
arity_handles = [
    Line2D(
        [0], [0],
        color=colors[a],
        linestyle="-",
        linewidth=4,
        label=f"{a}-way task "
              f"({results[a]['n_positive']}+ / {results[a]['n_negative']}- states)"
    )
    for a in arities
]

class_handles = [
    Line2D(
        [0], [0],
        color="black",
        linestyle="-",
        linewidth=3,
        label="true state in positive class"
    ),
    Line2D(
        [0], [0],
        color="black",
        linestyle="--",
        linewidth=3,
        label="true state in negative class"
    ),
]

legend1 = ax.legend(
    handles=arity_handles,
    title="Task arity",
    loc="center left",
    bbox_to_anchor=(1.02, 0.63),
    frameon=False,
)

legend2 = ax.legend(
    handles=class_handles,
    title="Stimulus class",
    loc="center left",
    bbox_to_anchor=(1.02, 0.30),
    frameon=False,
)

ax.add_artist(legend1)

# -----------------------------
# Annotation
# -----------------------------
# ax.text(
#     0.12,
#     0.08,
#     "Higher arity → smaller positive set\n"
#     "→ more concentration needed\n"
#     "to choose the positive class",
#     transform=ax.transAxes,
#     fontsize=14,
#     bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
# )

plt.tight_layout()
out_path1 = "experiments/simulations/04_lbt_recovery/results/plots/concentration.png"
plt.savefig(out_path1, dpi=150, bbox_inches="tight")
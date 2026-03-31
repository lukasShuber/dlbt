"""
Simulation 03 — belief distribution robustness analysis.

Loads results_frozen.pkl and produces:
  plot_robustness_cmse.png  — cMSE across distributions, all conditions
  plot_robustness_rho.png   — ρ    across distributions, all conditions

Run from repo root:
    python experiments/simulations/03_belief_distributions/analysis.py
"""

import pickle
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

import config as cfg

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
results_path = cfg.RESULTS_DIR / f"results_{cfg.RUN_TAG}.pkl"
if not results_path.exists():
    sys.exit(f"Results file not found: {results_path}\nRun run.py first.")

with open(results_path, "rb") as f:
    res = pickle.load(f)

plots_dir = cfg.RESULTS_DIR / "plots"
plots_dir.mkdir(exist_ok=True)

distributions = res["distributions"]
dist_labels   = res["dist_labels"]
dist_colors   = res["dist_colors"]
dlbt          = res["dlbt"]
slda          = res["slda"]

n_dist = len(distributions)
x      = np.arange(n_dist)
width  = 0.18   # bar width

# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------
COND_STYLES = [
    # (cond,   task_names_key,  color,      label,           hatch, model)
    ("train", "train", cfg.C_TRAIN, "DLBT train",     "",   "dlbt"),
    ("stim",  "stim",  cfg.C_STIM,  "DLBT stim gen",  "",   "dlbt"),
    ("task",  "task",  cfg.C_TASK,  "DLBT task gen",  "",   "dlbt"),
    ("joint", "joint", cfg.C_JOINT, "DLBT joint gen", "",   "dlbt"),
    ("train", "train", cfg.C_TRAIN, "SLDA train",     "//", "slda"),
    ("stim",  "stim",  cfg.C_STIM,  "SLDA stim gen",  "//", "slda"),
]

N_BARS = len(COND_STYLES)
offsets = np.linspace(-(N_BARS - 1) / 2, (N_BARS - 1) / 2, N_BARS) * width


def _plot_metric(ax, metric: str, ylabel: str):
    for i, (cond, _, color, label, hatch, model) in enumerate(COND_STYLES):
        data = dlbt[cond][metric] if model == "dlbt" else slda[cond][metric]
        mean = data.mean(axis=0)   # [n_dist]
        std  = data.std(axis=0)

        bars = ax.bar(x + offsets[i], mean, width,
                      color=color, alpha=0.85 if model == "dlbt" else 0.5,
                      hatch=hatch, edgecolor="white", linewidth=0.4,
                      label=label)
        ax.errorbar(x + offsets[i], mean, yerr=std,
                    fmt="none", color="black", capsize=2, linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([dist_labels[d] for d in distributions],
                       rotation=15, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    if metric == "cmse":
        ax.set_ylim(bottom=0)


# ---------------------------------------------------------------------------
# Plot 1 — cMSE
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4.5))
_plot_metric(ax, "cmse", "cMSE")
ax.set_title(
    f"Belief distribution robustness — cMSE  "
    f"({res['n_seeds'] if 'n_seeds' in res else cfg.N_SEEDS} seeds ± 1 SD)",
    fontsize=11,
)

handles = [
    mpatches.Patch(facecolor=color, alpha=0.85 if model == "dlbt" else 0.5,
                   hatch=hatch, label=label, edgecolor="gray")
    for _, _, color, label, hatch, model in COND_STYLES
]
ax.legend(handles=handles, fontsize=8, ncol=2, frameon=False,
          bbox_to_anchor=(1.01, 1), loc="upper left")

sns.despine(trim=True)
plt.tight_layout()
out = plots_dir / "plot_robustness_cmse.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()

# ---------------------------------------------------------------------------
# Plot 2 — ρ
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4.5))
_plot_metric(ax, "rho", "Spearman ρ")
ax.set_title(
    f"Belief distribution robustness — ρ  "
    f"({res['n_seeds'] if 'n_seeds' in res else cfg.N_SEEDS} seeds ± 1 SD)",
    fontsize=11,
)

ax.legend(handles=handles, fontsize=8, ncol=2, frameon=False,
          bbox_to_anchor=(1.01, 1), loc="upper left")

sns.despine(trim=True)
plt.tight_layout()
out = plots_dir / "plot_robustness_rho.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()

print("\nAll plots saved to", plots_dir)

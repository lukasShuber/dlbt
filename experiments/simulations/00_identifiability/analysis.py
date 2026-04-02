"""
Simulation 00 — identifiability analysis.

Produces:
  plot_identifiability.png  — 2×2 figure:
      row 0: recovered belief means vs GT belief means  (one col per mode)
      row 1: predicted p(right) vs true p(right)        (one col per mode)

Run from repo root:
    python experiments/simulations/00_identifiability/analysis.py
"""

import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import config as cfg

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
results_path = cfg.RESULTS_DIR / "results.pkl"
if not results_path.exists():
    sys.exit(f"Results not found: {results_path}\nRun run.py first.")

with open(results_path, "rb") as f:
    res = pickle.load(f)

plots_dir = cfg.RESULTS_DIR / "plots"
plots_dir.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 2×2 figure
#   col 0: one-hot      col 1: 4D latent
#   row 0: belief mean recovery
#   row 1: p(right) prediction
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(
    2, 2, figsize=(8, 8),
    gridspec_kw={"hspace": 0.38, "wspace": 0.12},
)

for col, mode in enumerate(cfg.FEATURE_MODES):
    d     = res[mode]
    color = cfg.MODE_COLORS[mode]

    # --- row 0: belief mean recovery ---
    ax = axes[0, col]
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1.2, zorder=0)
    ax.scatter(d["q_gt_flat"], d["q_rec_flat"],
               s=3, alpha=0.12, color=color, linewidths=0, rasterized=True)
    ax.set_title(cfg.MODE_LABELS[mode], fontsize=11, pad=5)
    ax.text(0.05, 0.95, f"$\\rho$ = {d['rho_q']:.3f}", transform=ax.transAxes,
            fontsize=10, va="top")
    ax.text(0.05, 0.87, f"MSE = {d['mse_q']:.5f}", transform=ax.transAxes,
            fontsize=10, va="top")
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(labelsize=9)
    if col == 0:
        ax.set_ylabel("Recovered  $\\hat{q}_k$", fontsize=11)

    # --- row 1: p(right) prediction ---
    ax = axes[1, col]
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1.2, zorder=0)
    ax.scatter(d["p_true"], d["p_pred"],
               s=4, alpha=0.20, color=color, linewidths=0, rasterized=True)
    ax.text(0.05, 0.95, f"$\\rho$ = {d['rho_p']:.3f}", transform=ax.transAxes,
            fontsize=10, va="top")
    ax.text(0.05, 0.87, f"MSE = {d['mse_p']:.5f}", transform=ax.transAxes,
            fontsize=10, va="top")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(labelsize=9)
    if col == 0:
        ax.set_ylabel("Predicted  $P(\\mathrm{right})$", fontsize=11)
    ax.set_xlabel("True  $P(\\mathrm{right})$", fontsize=11)

# Row labels on the right
for row, label in enumerate(["Belief mean recovery", "$P(\\mathrm{right})$ prediction"]):
    axes[row, 1].annotate(
        label, xy=(1.04, 0.5), xycoords="axes fraction",
        fontsize=10, rotation=-90, va="center", ha="left",
        color="gray",
    )

# Shared x label for row 0
fig.text(0.5, 0.535, "GT belief mean  $q_k$",
         ha="center", fontsize=11)

sns.despine(fig=fig, trim=False)
plt.tight_layout()

out = plots_dir / "plot_identifiability.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\nSummary:")
print(f"  {'Mode':<30}  {'ρ_q':>6}  {'MSE_q':>8}  {'ρ_p':>6}  {'MSE_p':>8}  {'epoch':>6}")
print(f"  {'-'*70}")
for mode in cfg.FEATURE_MODES:
    d = res[mode]
    print(f"  {cfg.MODE_LABELS[mode]:<30}  {d['rho_q']:>6.3f}  {d['mse_q']:>8.5f}"
          f"  {d['rho_p']:>6.3f}  {d['mse_p']:>8.5f}  {d['best_epoch']:>6d}")

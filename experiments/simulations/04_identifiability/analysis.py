"""
Simulation 04 — identifiability analysis.

Produces:
  plot_recovery_scatter.png  — 2-panel scatter of recovered vs GT belief means
                               (one panel per feature mode)

Run from repo root:
    python experiments/simulations/04_identifiability/analysis.py
"""

import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import spearmanr

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
# Recovery scatter — 2-panel (one-hot | latent)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.8), sharex=True, sharey=True,
                         gridspec_kw={"wspace": 0.10})

for ax, mode in zip(axes, cfg.FEATURE_MODES):
    d     = res[mode]
    x     = d["q_gt_flat"]
    y     = d["q_rec_flat"]
    color = cfg.MODE_COLORS[mode]
    rho   = d["rho"]
    mse   = d["mse"]

    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1.2, zorder=0)
    ax.scatter(x, y, s=3, alpha=0.12, color=color, linewidths=0, rasterized=True)

    ax.set_title(cfg.MODE_LABELS[mode], fontsize=11, pad=5)
    ax.text(0.05, 0.95, f"$\\rho$ = {rho:.3f}", transform=ax.transAxes,
            fontsize=10, va="top")
    ax.text(0.05, 0.87, f"MSE = {mse:.5f}", transform=ax.transAxes,
            fontsize=10, va="top")

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(labelsize=9)

fig.supxlabel("GT belief mean  $q_k$", fontsize=12, y=0.01)
fig.supylabel("Recovered belief mean  $\\hat{q}_k$", fontsize=12, x=0.01)

sns.despine(fig=fig, trim=False)
plt.tight_layout(rect=[0.05, 0.05, 1, 1])

out = plots_dir / "plot_recovery_scatter.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\nRecovery summary:")
print(f"  {'Mode':<20}  {'ρ':>6}  {'MSE':>10}  {'best_epoch':>10}")
print(f"  {'-'*52}")
for mode in cfg.FEATURE_MODES:
    d = res[mode]
    print(f"  {cfg.MODE_LABELS[mode]:<20}  {d['rho']:>6.4f}  "
          f"{d['mse']:>10.6f}  {d['best_epoch']:>10d}")

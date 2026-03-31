"""
Simulation 02 — data efficiency analysis.

Loads results_{tag}.pkl saved by run.py and produces:
  plot_efficiency_cmse_{tag}.png  — cMSE vs. budget
  plot_efficiency_rho_{tag}.png   — ρ vs. budget

Run from repo root:
    python experiments/simulations/02_data_efficiency/analysis.py [--tag frozen|attnpool]
"""

import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import config as cfg

# ---------------------------------------------------------------------------
# CLI: --tag frozen | attnpool  (default: cfg.RUN_TAG)
# ---------------------------------------------------------------------------
run_tag = cfg.RUN_TAG
for i, arg in enumerate(sys.argv[1:]):
    if arg == "--tag" and i + 1 < len(sys.argv) - 1:
        run_tag = sys.argv[i + 2]

results_path = cfg.RESULTS_DIR / f"results_{run_tag}.pkl"
if not results_path.exists():
    sys.exit(f"Results file not found: {results_path}\nRun run.py first.")

with open(results_path, "rb") as f:
    res = pickle.load(f)

plots_dir = cfg.RESULTS_DIR / "plots"
plots_dir.mkdir(exist_ok=True)

budgets = res["budgets"]
dlbt    = res["dlbt"]
slda    = res["slda"]

model_label = "DLBT (frozen)" if run_tag == "frozen" else "DLBT (attnpool)"

C_TRAIN, C_STIM, C_TASK, C_JOINT = cfg.C_TRAIN, cfg.C_STIM, cfg.C_TASK, cfg.C_JOINT

# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------
def _plot_metric(ax, metric: str, ylabel: str):
    """Draw all conditions for both models on ax."""
    x = np.array(budgets)

    # DLBT — 4 conditions, solid lines
    for cond, color, label in [
        ("train", C_TRAIN, f"{model_label} train"),
        ("stim",  C_STIM,  f"{model_label} stim gen"),
        ("task",  C_TASK,  f"{model_label} task gen"),
        ("joint", C_JOINT, f"{model_label} joint gen"),
    ]:
        vals = dlbt[cond][metric]          # [n_seeds, n_budgets]
        mean = vals.mean(axis=0)
        std  = vals.std(axis=0)
        ax.plot(x, mean, color=color, lw=2.0, label=label)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)

    # SLDA — 2 conditions, dashed lines
    for cond, color, label in [
        ("train", C_TRAIN, "SLDA train"),
        ("stim",  C_STIM,  "SLDA stim gen"),
    ]:
        vals = slda[cond][metric]
        mean = vals.mean(axis=0)
        std  = vals.std(axis=0)
        ax.plot(x, mean, color=color, lw=2.0, ls="--", label=label)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.10)

    ax.set_xscale("log")
    ax.set_xticks(budgets)
    labels = [f"{b:,}" if b < 1_000_000 else "1M" for b in budgets]
    ax.set_xticklabels(labels)
    ax.set_xlabel("Total training trials", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    if metric == "cmse":
        ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, ncol=2, frameon=False)


# ---------------------------------------------------------------------------
# Plot 1 — cMSE
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
_plot_metric(ax, "cmse", "cMSE")
ax.set_title(f"Data efficiency — cMSE  ({run_tag}, {len(res['seeds'])} seeds ± 1 SD)",
             fontsize=11)
sns.despine(trim=True)
plt.tight_layout()
out = plots_dir / f"plot_efficiency_cmse_{run_tag}.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()

# ---------------------------------------------------------------------------
# Plot 2 — ρ
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
_plot_metric(ax, "rho", "Spearman ρ")
ax.set_title(f"Data efficiency — ρ  ({run_tag}, {len(res['seeds'])} seeds ± 1 SD)",
             fontsize=11)
sns.despine(trim=True)
plt.tight_layout()
out = plots_dir / f"plot_efficiency_rho_{run_tag}.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()

print("\nAll plots saved to", plots_dir)

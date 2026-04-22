"""
02_data_efficiency/analysis.py — plots for the data-efficiency sweep.

Generated figures:
  plot_01_cmse_vs_budget.png   — cMSE−NF vs trial budget, one line per region
  plot_02_curves_<budget>.png  — learning curves at each budget level

Run from repo root:
    python experiments/behavior/run0/02_data_efficiency/analysis.py
"""

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

plots_dir = cfg.RESULTS_DIR / "plots"
plots_dir.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------
results_path = cfg.RESULTS_DIR / f"data_efficiency_{cfg.RUN_TAG}.pkl"
if not results_path.exists():
    raise FileNotFoundError(f"No results found at {results_path}. Run run.py first.")

with open(results_path, "rb") as f:
    summary = pickle.load(f)

results   = summary["results"]
nfs       = summary["noise_floors"]
rg_joint  = summary["random_guesser_joint_gen_cmse_net"]
n_pool    = summary["n_pool"]

# Ordered budget labels (full comes last)
# Build (x, label, y) for every available budget, sorted by x (actual trial count).
# "full" is placed at n_pool — wherever that lands on the axis.
all_points = []
for label, res in results.items():
    x = n_pool if label == "full" else int(label)
    all_points.append((x, label, res["joint_gen_cmse_net"]))
all_points.sort(key=lambda p: p[0])

x_all     = [p[0] for p in all_points]
lab_all   = [p[1] for p in all_points]
y_all     = [p[2] for p in all_points]

# Split at "full": solid up to (and including) "full", dashed beyond.
full_idx  = next((i for i, p in enumerate(all_points) if p[1] == "full"), len(all_points) - 1)
x_solid   = x_all[:full_idx + 1]
y_solid   = y_all[:full_idx + 1]
x_dashed  = x_all[full_idx:]       # overlaps at "full" so the lines connect
y_dashed  = y_all[full_idx:]

# ---------------------------------------------------------------------------
# Plot 01 — cMSE−NF vs trial budget
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))

ax.plot(x_solid, y_solid, "o-",  color=cfg.C_JOINT, lw=1.8, ms=6,
        label="joint gen")

# Random guesser line on joint_gen only
ax.axhline(rg_joint, ls="--", color=cfg.C_JOINT, alpha=0.5, lw=1.2,
           label="random (joint gen)")

ax.axhline(0, ls=":", color="gray", lw=0.8, alpha=0.6)

ax.set_xscale("log")
ax.set_xlabel("Trial budget", fontsize=11)
ax.set_ylabel("cMSE − noise floor", fontsize=11)
ax.set_title("Data efficiency: DLBT generalisation vs trial budget", fontsize=11)

# Tick at every solid point; use the label string for each
ax.set_xticks(x_solid)
ax.set_xticklabels(lab_all[:full_idx + 1], fontsize=9)

ax.legend(fontsize=9, frameon=False)
sns.despine(trim=True)
plt.tight_layout()
out = plots_dir / "plot_01_cmse_vs_budget.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()

# ---------------------------------------------------------------------------
# Plot 02 — learning curves per budget
# ---------------------------------------------------------------------------
for label in lab_all:
    res_b  = results[label]
    curves = res_b["curves"]
    epochs = range(len(curves["train_mses"]))

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(epochs, curves["train_mses"],  color=cfg.C_TRAIN,  label="train",     lw=1.2)
    ax.plot(epochs, curves["eval_mses"],   color=cfg.C_EVAL,   label="eval",      lw=1.2)
    if curves.get("stim_mses"):
        ax.plot(epochs, curves["stim_mses"],   color=cfg.C_STIM,   label="stim gen",  lw=1.0, alpha=0.7)
    if curves.get("task_mses"):
        ax.plot(epochs, curves["task_mses"],   color=cfg.C_TASK,   label="task gen",  lw=1.0, alpha=0.7)
    if curves.get("joint_mses"):
        ax.plot(epochs, curves["joint_mses"],  color=cfg.C_JOINT,  label="joint gen", lw=1.0, alpha=0.7)

    ax.axvline(res_b["best_epoch"], ls=":", color="gray", lw=0.8)

    for key, color in [("eval", cfg.C_EVAL), ("stim_gen", cfg.C_STIM),
                       ("task_gen", cfg.C_TASK), ("joint_gen", cfg.C_JOINT)]:
        if key in nfs:
            ax.axhline(nfs[key], ls="--", color=color, alpha=0.35, lw=1)

    ax.set(xlabel="epoch", ylabel="cMSE",
           title=f"Budget = {label}  (trials={res_b['n_trials']}, cells={res_b['n_cells']})")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, frameon=False)
    sns.despine(trim=True)
    plt.tight_layout()
    out = plots_dir / f"plot_02_curves_budget{label}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()

print("\nAll plots saved to", plots_dir)

"""
Simulation 01 — analysis and plots.

Loads results saved by run.py and generates four figures:
  plot_02_curves.png        — DLBT learning curves (NLL + cMSE)
  plot_03_summary.png       — 6-panel pred-vs-true scatter
  plot_04_per_task_dlbt.png — per-task scatter grid (DLBT)
  plot_05_per_task_slda.png — per-task scatter grid (SLDA)

Run from repo root:
    python experiments/simulations/01_four_dim_generalization/analysis.py [--tag frozen|attnpool]
"""

import math
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

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

model_label    = res["model_label"]
phase_boundary = res["phase_boundary"]
best_epoch     = res["best_epoch"]
noise_floor    = res["noise_floor"]
curves         = res["curves"]
dlbt_train     = res["dlbt_train"]
dlbt_stim      = res["dlbt_stim"]
dlbt_task      = res["dlbt_task"]
dlbt_joint     = res["dlbt_joint"]
slda_train     = res["slda_train"]
slda_stim      = res["slda_stim"]

has_phase2 = phase_boundary < len(curves["train_nlls"]) - 1

C_TRAIN, C_STIM, C_TASK, C_JOINT = cfg.C_TRAIN, cfg.C_STIM, cfg.C_TASK, cfg.C_JOINT

# ---------------------------------------------------------------------------
# Plot 2 — learning curves
# ---------------------------------------------------------------------------
epochs = range(len(curves["train_nlls"]))

fig, (ax_nll, ax_mse) = plt.subplots(1, 2, figsize=(11, 3.8))
for ax, tr, vl, tg, jg, ylabel in [
    (ax_nll,
     curves["train_nlls"], curves["val_nlls"],
     curves["task_nlls"],  curves["joint_nlls"], "NLL"),
    (ax_mse,
     curves["train_mses"], curves["val_mses"],
     curves["task_mses"],  curves["joint_mses"], "cMSE"),
]:
    ax.plot(epochs, tr, color=C_TRAIN, label="train",    lw=1.2)
    ax.plot(epochs, vl, color=C_STIM,  label="stim gen", lw=1.2)
    ax.plot(epochs, tg, color=C_TASK,  label="task gen", lw=1.2)
    ax.plot(epochs, jg, color=C_JOINT, label="joint gen",lw=1.2)
    ax.axvline(best_epoch, ls=":", color="gray", lw=0.8)
    if has_phase2:
        ax.axvline(phase_boundary, ls="--", color="black", lw=0.8, alpha=0.5)
        ax.text(phase_boundary + 1, 0.98, "phase 2", fontsize=7,
                va="top", transform=ax.get_xaxis_transform(), color="black", alpha=0.6)
    ax.set(ylabel=ylabel, xlabel="epoch", title=f"{model_label} — {ylabel}")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)

ax_mse.axhline(noise_floor, ls="--", color=C_TRAIN, alpha=0.4, lw=1)
sns.despine(trim=True)
plt.tight_layout()
out = plots_dir / f"plot_02_curves_{run_tag}.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()

# ---------------------------------------------------------------------------
# Plot 3 — 6-panel summary scatter
# ---------------------------------------------------------------------------
def _summary_scatter(ax, pt: dict, task_names: list, color: str,
                     title: str, mc_n=None):
    preds = np.concatenate([pt[t]["pred"] for t in task_names if t in pt])
    trues = np.concatenate([pt[t]["true"] for t in task_names if t in pt])
    raw   = float(np.mean((preds - trues) ** 2))
    cmse  = raw - float(np.mean(preds * (1 - preds))) / (mc_n - 1) if mc_n else raw
    from scipy.stats import spearmanr
    rho, _ = spearmanr(preds, trues)
    ax.plot([0, 1], [0, 1], ls=":", color="gray", lw=0.8, zorder=0)
    ax.scatter(preds, trues, alpha=0.3, s=8, color=color, linewidths=0)
    ax.set(title=f"{title}\ncMSE={cmse:.4f}   ρ={rho:.3f}",
           xlim=(-0.05, 1.05), ylim=(-0.05, 1.05))
    ax.tick_params(labelsize=8)


panels = [
    # col 0: SLDA
    (slda_train, cfg.TRAIN_TASKS, C_TRAIN, "SLDA — Train",              None,     0, 0),
    (slda_stim,  cfg.TRAIN_TASKS, C_STIM,  "SLDA — Stim gen",           None,     1, 0),
    # col 1: DLBT train / stim
    (dlbt_train, cfg.TRAIN_TASKS, C_TRAIN, f"{model_label} — Train",    cfg.N_MC, 0, 1),
    (dlbt_stim,  cfg.TRAIN_TASKS, C_STIM,  f"{model_label} — Stim gen", cfg.N_MC, 1, 1),
    # col 2: generalization
    (dlbt_task,  cfg.VAL_TASKS,   C_TASK,  f"{model_label} — Task gen", cfg.N_MC, 0, 2),
    (dlbt_joint, cfg.VAL_TASKS,   C_JOINT, f"{model_label} — Joint gen",cfg.N_MC, 1, 2),
]

fig, axes = plt.subplots(2, 3, figsize=(10, 7), sharex=True, sharey=True,
                         gridspec_kw={"hspace": 0.45, "wspace": 0.12})
for pt, task_names, color, title, mc_n, row, col in panels:
    ax = axes[row, col]
    _summary_scatter(ax, pt, task_names, color, title, mc_n)
    if col == 0:
        ax.set_ylabel("True P(right)", fontsize=9)
    if row == 1:
        ax.set_xlabel("Predicted P(right)", fontsize=9)

sns.despine(fig=fig, trim=True)
out = plots_dir / f"plot_03_summary_{run_tag}.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()

# ---------------------------------------------------------------------------
# Plot 4 — per-task DLBT grid
# ---------------------------------------------------------------------------
ALL_TASKS = cfg.TRAIN_TASKS + cfg.VAL_TASKS
N_COLS    = 8
N_ROWS    = math.ceil(len(ALL_TASKS) / N_COLS)

fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(N_COLS * 2.0, N_ROWS * 2.2),
                         sharex=True, sharey=True,
                         gridspec_kw={"hspace": 0.55, "wspace": 0.08})
for ax in axes.flat[len(ALL_TASKS):]:
    ax.set_visible(False)

for idx, (ax, task_name) in enumerate(zip(axes.flat, ALL_TASKS)):
    ax.plot([0, 1], [0, 1], ls=":", color="gray", lw=0.7, zorder=0)
    is_val = task_name in cfg.VAL_TASKS
    if not is_val:
        for pt, color in [(dlbt_train, C_TRAIN), (dlbt_stim, C_STIM)]:
            if task_name in pt:
                d = pt[task_name]
                ax.scatter(d["pred"], d["true"], alpha=0.5, s=5, color=color, linewidths=0)
        ax.text(0.05, 0.93, f"ρ={dlbt_train.get(task_name, {}).get('rho', float('nan')):.2f}",
                transform=ax.transAxes, fontsize=6, color=C_TRAIN, va="top")
        ax.text(0.05, 0.78, f"ρ={dlbt_stim.get(task_name, {}).get('rho', float('nan')):.2f}",
                transform=ax.transAxes, fontsize=6, color=C_STIM, va="top")
    else:
        for pt, color in [(dlbt_task, C_TASK), (dlbt_joint, C_JOINT)]:
            if task_name in pt:
                d = pt[task_name]
                ax.scatter(d["pred"], d["true"], alpha=0.5, s=5, color=color, linewidths=0)
        ax.text(0.05, 0.93, f"ρ={dlbt_task.get(task_name, {}).get('rho', float('nan')):.2f}",
                transform=ax.transAxes, fontsize=6, color=C_TASK, va="top")
        ax.text(0.05, 0.78, f"ρ={dlbt_joint.get(task_name, {}).get('rho', float('nan')):.2f}",
                transform=ax.transAxes, fontsize=6, color=C_JOINT, va="top")

    ax.set_title(task_name.replace("_and_", " & ").replace("_", "/"), fontsize=7, pad=2)
    row, col = divmod(idx, N_COLS)
    if row == N_ROWS - 1:
        ax.set_xlabel("Pred", fontsize=7)
    if col == 0:
        ax.set_ylabel("True", fontsize=7)
    ax.set(xlim=(-0.05, 1.05), ylim=(-0.05, 1.05))
    ax.tick_params(labelsize=5)

fig.legend(handles=[
    Line2D([0],[0], marker="o", color="w", markerfacecolor=c, markersize=5, label=l)
    for c, l in [(C_TRAIN,"train"),(C_STIM,"stim gen"),(C_TASK,"task gen"),(C_JOINT,"joint gen")]
], loc="lower right", bbox_to_anchor=(1.0, 0.0), fontsize=7, frameon=False, ncol=2)
fig.text(0.5, -0.01, "Predicted P(right)", ha="center", fontsize=9)
fig.text(-0.01, 0.5, "True P(right)", va="center", rotation="vertical", fontsize=9)
sns.despine(fig=fig, trim=True)
out = plots_dir / f"plot_04_per_task_dlbt_{run_tag}.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()

# ---------------------------------------------------------------------------
# Plot 5 — per-task SLDA grid
# ---------------------------------------------------------------------------
N_COLS_S = 8
N_ROWS_S = math.ceil(len(cfg.TRAIN_TASKS) / N_COLS_S)

fig, axes = plt.subplots(N_ROWS_S, N_COLS_S, figsize=(N_COLS_S * 2.0, N_ROWS_S * 2.2),
                         sharex=True, sharey=True,
                         gridspec_kw={"hspace": 0.55, "wspace": 0.08})
for ax in axes.flat[len(cfg.TRAIN_TASKS):]:
    ax.set_visible(False)

for idx, (ax, task_name) in enumerate(zip(axes.flat, cfg.TRAIN_TASKS)):
    ax.plot([0, 1], [0, 1], ls=":", color="gray", lw=0.7, zorder=0)
    for pt, color in [(slda_train, C_TRAIN), (slda_stim, C_STIM)]:
        if task_name in pt:
            d = pt[task_name]
            ax.scatter(d["pred"], d["true"], alpha=0.5, s=5, color=color,
                       linewidths=0, marker="s")
    ax.text(0.05, 0.93, f"ρ={slda_train.get(task_name, {}).get('rho', float('nan')):.2f}",
            transform=ax.transAxes, fontsize=6, color=C_TRAIN, va="top")
    ax.text(0.05, 0.78, f"ρ={slda_stim.get(task_name, {}).get('rho', float('nan')):.2f}",
            transform=ax.transAxes, fontsize=6, color=C_STIM, va="top")
    ax.set_title(task_name.replace("_and_", " & ").replace("_", "/"), fontsize=7, pad=2)
    row, col = divmod(idx, N_COLS_S)
    if row == N_ROWS_S - 1:
        ax.set_xlabel("Pred", fontsize=7)
    if col == 0:
        ax.set_ylabel("True", fontsize=7)
    ax.set(xlim=(-0.05, 1.05), ylim=(-0.05, 1.05))
    ax.tick_params(labelsize=5)

fig.legend(handles=[
    Line2D([0],[0], marker="s", color="w", markerfacecolor=c, markersize=5, label=l)
    for c, l in [(C_TRAIN,"train"),(C_STIM,"stim gen")]
], loc="lower right", bbox_to_anchor=(1.0, 0.0), fontsize=7, frameon=False)
fig.text(0.5, -0.01, "Predicted P(right)", ha="center", fontsize=9)
fig.text(-0.01, 0.5, "True P(right)", va="center", rotation="vertical", fontsize=9)
sns.despine(fig=fig, trim=True)
out = plots_dir / f"plot_05_per_task_slda_{run_tag}.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()

print("\nAll plots saved to", plots_dir)

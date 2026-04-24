"""
run1/02_data_efficiency/analysis.py — plots for the data-efficiency sweep.

Generated figures:
  plot_01_cmse_vs_budget_{tag}.png   — cMSE−NF vs trial budget (all regions;
                                       solid = h=0, dashed = h_n corrected)
  plot_02_curves_{tag}_{budget}.png  — learning curves at each budget level

Run from repo root:
    python experiments/behavior/run1/02_data_efficiency/analysis.py
"""

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
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
    raise FileNotFoundError(f"No results at {results_path}. Run run.py first.")

with open(results_path, "rb") as f:
    summary = pickle.load(f)

results   = summary["results"]
nfs       = summary["noise_floors"]
n_pool    = summary["n_pool"]
has_corr  = summary.get("threshold_correction", False)

# ---------------------------------------------------------------------------
# Sort budgets by trial count
# ---------------------------------------------------------------------------
all_points = []
for label, res in results.items():
    x = n_pool if label == "full" else int(label)
    all_points.append((x, label, res))
all_points.sort(key=lambda p: p[0])

x_all   = [p[0] for p in all_points]
lab_all = [p[1] for p in all_points]

full_idx = next((i for i, p in enumerate(all_points) if p[1] == "full"),
                len(all_points) - 1)

# ---------------------------------------------------------------------------
# Plot 01 — cMSE−NF vs trial budget
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.5))

region_cfg = [
    ("train_cmse_net",    cfg.C_TRAIN, "train"),
    ("stim_gen_cmse_net", cfg.C_STIM,  "stim gen"),
    ("task_gen_cmse_net", cfg.C_TASK,  "task gen"),
    ("joint_gen_cmse_net",cfg.C_JOINT, "joint gen"),
]
if has_corr:
    region_cfg += [
        ("task_gen_h_cmse_net",  cfg.C_TASK,  "task gen (hₙ)"),
        ("joint_gen_h_cmse_net", cfg.C_JOINT, "joint gen (hₙ)"),
    ]

for key, color, label in region_cfg:
    y_vals = [p[2].get(key, float("nan")) for p in all_points]
    ls  = "-" if "(hₙ)" not in label else "--"
    lw  = 1.8 if "(hₙ)" not in label else 1.4
    ms  = 6   if "(hₙ)" not in label else 5
    mrk = "o" if "(hₙ)" not in label else "s"
    ax.plot(x_all[:full_idx + 1], y_vals[:full_idx + 1],
            f"{mrk}{ls}", color=color, lw=lw, ms=ms, label=label)

ax.axhline(0, ls=":", color="gray", lw=0.8, alpha=0.6)
ax.set_xscale("log")
ax.set_xlabel("Trial budget", fontsize=11)
ax.set_ylabel("cMSE − noise floor", fontsize=11)
ax.set_title(f"Data efficiency — DLBT  [{cfg.RUN_TAG}]", fontsize=11)

ax.set_xticks(x_all[:full_idx + 1])
ax.set_xticklabels(lab_all[:full_idx + 1], fontsize=9)
ax.legend(fontsize=8, frameon=False, ncol=2)
sns.despine(trim=True)
plt.tight_layout()
out = plots_dir / f"plot_01_cmse_vs_budget_{cfg.RUN_TAG}.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()

# ---------------------------------------------------------------------------
# Plot 02 — learning curves per budget
# ---------------------------------------------------------------------------
for x, label, res in all_points:
    curves = res.get("curves")
    if not curves:
        continue
    epochs = range(len(curves["train_mses"]))

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(epochs, curves["train_mses"], color=cfg.C_TRAIN, label="train",    lw=1.2)
    ax.plot(epochs, curves["eval_mses"],  color=cfg.C_EVAL,  label="eval",     lw=1.2)
    if curves.get("stim_mses"):
        ax.plot(epochs, curves["stim_mses"],  color=cfg.C_STIM,  label="stim gen", lw=1.0, alpha=0.7)
    if curves.get("task_mses"):
        ax.plot(epochs, curves["task_mses"],  color=cfg.C_TASK,  label="task gen", lw=1.0, alpha=0.7)
    if curves.get("joint_mses"):
        ax.plot(epochs, curves["joint_mses"], color=cfg.C_JOINT, label="joint gen",lw=1.0, alpha=0.7)

    ax.axvline(res["best_epoch"], ls=":", color="gray", lw=0.8)

    for nf_key, color in [("eval",      cfg.C_EVAL),
                           ("stim_gen",  cfg.C_STIM),
                           ("task_gen",  cfg.C_TASK),
                           ("joint_gen", cfg.C_JOINT)]:
        if nf_key in nfs:
            ax.axhline(nfs[nf_key], ls="--", color=color, alpha=0.35, lw=1)

    ax.set(xlabel="epoch", ylabel="cMSE",
           title=f"Budget = {label}  (trials={res['n_trials']}, cells={res['n_cells']})")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, frameon=False)
    sns.despine(trim=True)
    plt.tight_layout()
    out = plots_dir / f"plot_02_curves_{cfg.RUN_TAG}_budget{label}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()

print("\nAll plots saved to", plots_dir)

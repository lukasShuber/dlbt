"""
run1/02_data_efficiency/analysis.py — plots for the data-efficiency sweep.

Generated figures:
  plot_01_cmse_vs_budget_{tag}.png   — cMSE−NF vs trial budget (all regions;
                                       solid = h=0, dashed = h_n corrected)
  plot_02_curves_{tag}_{budget}.png  — learning curves at each budget level

Run from repo root:
    python experiments/behavior/run1/02_data_efficiency/analysis.py
"""

import math
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

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

# ---------------------------------------------------------------------------
# Plot 03 — per-task scatter for joint generalisation (one figure per budget)
# ---------------------------------------------------------------------------
ARITY_COLOR = {1: "#2a6fb5", 2: "#43AA8B", 3: "#E76F51", 4: "#9B5DE5"}
N_COLS      = 12

val_tasks = sorted(
    summary.get("val_tasks", []),
    key=lambda t: (t.count("_and_") + 1, t),
)


def _arity(t):
    return t.count("_and_") + 1


def _label(t):
    return t.replace("_and_", " & ").replace("_", "/")


def _true_sem(true_vals, totals):
    safe = np.clip(totals, 1, None)
    sem  = np.sqrt(np.clip(true_vals * (1 - true_vals), 0, None) / safe)
    sem[totals <= 0] = 0
    return sem


def _cmse_nf_joint(pred, true, totals, n_mc):
    raw = float(np.mean((pred - true) ** 2))
    if n_mc > 1:
        raw -= float(np.mean(pred * (1 - pred))) / (n_mc - 1)
    nf = float(np.mean(
        true[totals > 1] * (1 - true[totals > 1]) / (totals[totals > 1] - 1)
    )) if (totals > 1).any() else 0.0
    return raw - nf


for x, label, res in all_points:
    preds_b    = res.get("preds", {})
    joint_preds = preds_b.get("joint_gen", {})
    if not joint_preds:
        continue

    present = [t for t in val_tasks if t in joint_preds]
    if not present:
        continue

    # pooled metrics
    all_pred   = np.concatenate([joint_preds[t]["pred"]   for t in present])
    all_true   = np.concatenate([joint_preds[t]["true"]   for t in present])
    all_totals = np.concatenate([joint_preds[t]["totals"] for t in present])
    valid      = all_totals > 0
    if valid.sum() >= 2:
        rho_r, _ = spearmanr(all_pred[valid], all_true[valid])
        mse_r    = _cmse_nf_joint(all_pred[valid], all_true[valid],
                                   all_totals[valid], cfg.N_MC)
    else:
        rho_r = mse_r = float("nan")

    n_tasks = len(present)
    n_rows  = math.ceil(n_tasks / N_COLS)

    fig_t, axes_t = plt.subplots(
        n_rows, N_COLS,
        figsize=(N_COLS * 2.8, n_rows * 2.8),
        gridspec_kw={"hspace": 0.55, "wspace": 0.20},
    )
    axes_flat = np.atleast_2d(axes_t).flatten()
    for ax in axes_flat[n_tasks:]:
        ax.set_visible(False)

    for i, (ax, task_name) in enumerate(zip(axes_flat, present)):
        d     = joint_preds[task_name]
        v     = d["totals"] > 0
        pm    = d["pred"][v]
        tv    = d["true"][v]
        tot   = d["totals"][v]
        ts    = _true_sem(tv, tot)
        arity = _arity(task_name)
        color = ARITY_COLOR.get(arity, "#555")

        ax.plot([0, 1], [0, 1], ls=":", color="gray", lw=0.7, zorder=0)
        ax.errorbar(pm, tv, yerr=ts,
                    fmt="o", ms=4, alpha=0.85, color=color,
                    elinewidth=0.5, capsize=0, linewidth=0, zorder=2)

        if v.sum() >= 2:
            rc, _ = spearmanr(pm, tv)
            mc_v  = _cmse_nf_joint(pm, tv, tot, cfg.N_MC)
            ax.text(0.05, 0.97, f"{arity}-way  ρ={rc:.2f}",
                    transform=ax.transAxes, fontsize=7, color=color,
                    va="top")
            ax.text(0.05, 0.81, f"m={mc_v:.3f}",
                    transform=ax.transAxes, fontsize=7, color="gray",
                    va="top")

        ax.set_title(_label(task_name), fontsize=8, pad=3, color=color)
        ax.set(xlim=(-0.05, 1.05), ylim=(-0.05, 1.05))
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(labelsize=6)

        row_i, col_i = divmod(i, N_COLS)
        if row_i == n_rows - 1 or i >= n_tasks - N_COLS:
            ax.set_xlabel("Pred", fontsize=7)
        if col_i == 0:
            ax.set_ylabel("Human", fontsize=7)

    handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=c, markersize=5, label=f"{a}-way")
        for a, c in ARITY_COLOR.items() if a > 1
    ]
    fig_t.legend(handles=handles, loc="lower right",
                 bbox_to_anchor=(1.0, 0.0), fontsize=7,
                 frameon=False, ncol=len(handles))

    fig_t.suptitle(
        f"Joint gen — per task  [budget={label}]  [{cfg.RUN_TAG}]\n"
        f"ρ={rho_r:.3f}   cMSE-NF={mse_r:+.4f}",
        fontsize=9, y=1.01,
    )
    sns.despine(fig=fig_t, trim=True)
    plt.tight_layout()
    out = plots_dir / f"plot_03_pertask_joint_{cfg.RUN_TAG}_budget{label}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig_t)

print("\nAll plots saved to", plots_dir)

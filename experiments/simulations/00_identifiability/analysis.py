"""
Analysis for simulation 00 — identifiability experiment.

Produces:
  plot_01_alpha_recovery.png  — scatter true alpha* vs predicted alpha (all images × K states)
  plot_02_learning_curves.png — training / val NLL and MSE over epochs
  plot_03_summary.png         — pred-vs-true P(right) scatter, 4 generalization conditions
  plot_04_per_task_dlbt.png   — per-task scatter grid (DLBT)
  plot_05_per_task_slda.png   — per-task scatter grid (SLDA)

Run from repo root:
    python experiments/simulations/00_identifiability/analysis.py [--tag frozen|attnpool]
"""

import argparse
import math
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

import config as cfg

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--tag", default=None, choices=["frozen", "attnpool"],
                    help="Filter to a specific results pkl (default: all available)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Discover available results
# ---------------------------------------------------------------------------
candidates = [
    cfg.RESULTS_DIR / "results_frozen.pkl",
    cfg.RESULTS_DIR / "results_attnpool.pkl",
]
if args.tag:
    candidates = [cfg.RESULTS_DIR / f"results_{args.tag}.pkl"]

available = []
for p in candidates:
    if not p.exists():
        continue
    with open(p, "rb") as _f:
        _r = pickle.load(_f)
    if "dlbt" in _r:
        available.append(p)
    else:
        print(f"Skipping {p.name} — old format (missing 'dlbt' key)")

if not available:
    raise FileNotFoundError("No valid results pkl found. Run run.py first.")

# ===========================================================================
# Loop over results files
# ===========================================================================
for results_path in available:
    run_tag = results_path.stem[len("results_"):]
    print(f"\n=== Processing: {results_path.name} (run_tag={run_tag}) ===")

    with open(results_path, "rb") as f:
        res = pickle.load(f)

    model_label    = res["model_label"]
    phase_boundary = res["phase_boundary"]
    best_epoch     = res["best_epoch"]
    noise_floor    = res["noise_floor"]
    curves         = res["curves"]
    dlbt           = res["dlbt"]
    slda           = res["slda"]
    n_seeds        = res.get("n_seeds", 1)
    n_trials       = res.get("n_trials", cfg.N_TRIALS)
    true_alphas    = res["true_alphas"]    # {uid: [K]}
    alpha_preds    = res["alpha_preds"]   # {uid: [n_seeds, K]}
    train_uids     = res["train_uids"]
    test_uids      = res["test_uids"]

    plots_dir = cfg.RESULTS_DIR / f"plots_{run_tag}"
    plots_dir.mkdir(exist_ok=True)

    has_phase2 = phase_boundary < len(curves["train_nlls"]) - 1

    C_TRAIN = cfg.C_TRAIN
    C_STIM  = cfg.C_STIM
    C_TASK  = cfg.C_TASK
    C_JOINT = cfg.C_JOINT

    # -----------------------------------------------------------------------
    # Plot 1 — Alpha recovery
    # -----------------------------------------------------------------------
    # Scatter true alpha*(uid, k) vs predicted alpha mean across seeds.
    # One panel per state k (K=16), colored by train/test split.
    # -----------------------------------------------------------------------
    uids_sorted = sorted(true_alphas.keys())
    true_mat  = np.array([true_alphas[u]               for u in uids_sorted])  # [N, K]
    pred_mat  = np.array([alpha_preds[u].mean(axis=0)  for u in uids_sorted])  # [N, K]
    pred_sem  = np.array([alpha_preds[u].std(axis=0) / np.sqrt(n_seeds)
                          for u in uids_sorted])                                 # [N, K]
    is_train  = np.array([u in train_uids for u in uids_sorted])

    fig, axes = plt.subplots(4, 4, figsize=(10, 10),
                             sharex=False, sharey=False,
                             gridspec_kw={"hspace": 0.55, "wspace": 0.35})

    for k, ax in enumerate(axes.flat):
        t = true_mat[:, k]
        p = pred_mat[:, k]
        s = pred_sem[:, k]

        lo = min(t.min(), p.min()) * 0.9
        hi = max(t.max(), p.max()) * 1.1
        ax.plot([lo, hi], [lo, hi], ls="--", color="gray", lw=0.8, zorder=0)

        for mask, color, label in [
            (is_train,  C_TRAIN, "train"),
            (~is_train, C_STIM,  "test"),
        ]:
            ax.errorbar(t[mask], p[mask],
                        yerr=s[mask],
                        fmt="o", ms=3, alpha=0.3, color=color,
                        elinewidth=0.4, capsize=0, linewidth=0)

        rho, _ = spearmanr(t, p)
        ax.set_title(f"state {k}\nρ={rho:.2f}", fontsize=7, pad=3)
        ax.tick_params(labelsize=6)

    fig.supxlabel("True α*", fontsize=11)
    fig.supylabel("Predicted α (DLBT mean)", fontsize=11, x=0.02)
    fig.suptitle(f"Alpha recovery — {model_label}", fontsize=12, y=1.01)

    legend_handles = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor=C_TRAIN, ms=5, label="train images"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor=C_STIM,  ms=5, label="test images"),
    ]
    axes.flat[0].legend(handles=legend_handles, fontsize=7, loc="upper left")

    plt.tight_layout()
    out = plots_dir / f"plot_01_alpha_recovery_{run_tag}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()

    # -----------------------------------------------------------------------
    # Plot 2 — Learning curves
    # -----------------------------------------------------------------------
    epochs = np.arange(len(curves["train_nlls"]))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, (key_tr, key_va, key_ta, key_jo), ylabel in zip(
        axes,
        [("train_nlls", "val_nlls",  "task_nlls",  "joint_nlls"),
         ("train_mses", "val_mses",  "task_mses",  "joint_mses")],
        ["NLL", "MSE"],
    ):
        ax.plot(epochs, curves[key_tr], color=C_TRAIN, lw=1.2, label="train")
        ax.plot(epochs, curves[key_va], color=C_STIM,  lw=1.2, label="stim gen")
        ax.plot(epochs, curves[key_ta], color=C_TASK,  lw=1.2, label="task gen")
        ax.plot(epochs, curves[key_jo], color=C_JOINT, lw=1.2, label="joint gen")
        if has_phase2:
            ax.axvline(phase_boundary, ls="--", color="black", lw=0.8, alpha=0.5)
        ax.axvline(best_epoch, ls=":", color="black", lw=0.8, alpha=0.5)
        if ylabel == "MSE":
            ax.axhline(noise_floor, ls="--", color=C_TRAIN, alpha=0.4, lw=1,
                       label="noise floor")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)

    fig.suptitle(f"Learning curves — {model_label}", fontsize=12)
    plt.tight_layout()
    out = plots_dir / f"plot_02_learning_curves_{run_tag}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()

    # -----------------------------------------------------------------------
    # Plot 3 — Summary scatter (pred vs true P(right))
    # -----------------------------------------------------------------------
    from scipy.stats import spearmanr

    def _summary_scatter(ax, pt, task_names, color, marker, title,
                         mc_n=None, n_seeds=1, n_trials=100):
        all_preds = np.concatenate(
            [pt[t]["pred"] for t in task_names if t in pt], axis=-1
        )
        all_trues = np.concatenate(
            [pt[t]["true"] for t in task_names if t in pt]
        )
        if all_preds.ndim == 2:
            pred_mean = all_preds.mean(axis=0)
            pred_sem  = all_preds.std(axis=0) / np.sqrt(n_seeds)
        else:
            pred_mean = all_preds
            pred_sem  = np.zeros_like(pred_mean)

        true_sem = np.sqrt(np.clip(all_trues * (1 - all_trues), 0, None) / n_trials)
        raw   = float(np.mean((pred_mean - all_trues) ** 2))
        cmse  = raw - float(np.mean(pred_mean * (1 - pred_mean))) / (mc_n - 1) if mc_n else raw
        rho, _ = spearmanr(pred_mean, all_trues)

        ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1.2, zorder=0)
        ax.errorbar(pred_mean, all_trues,
                    xerr=pred_sem, yerr=true_sem,
                    fmt=marker, ms=4, alpha=0.1, color=color,
                    elinewidth=0.5, capsize=0, linewidth=0)
        ax.set_title(f"{title}\ncMSE={cmse:.4f}   ρ={rho:.3f}", fontsize=10, pad=4)
        ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.tick_params(labelsize=9)

    # 4 conditions × 2 models = 8 panels, arranged 2×4
    panels = [
        # (pred_dict, task_names, color, marker, title, mc_n, row, col)
        (dlbt["train"], cfg.TRAIN_TASKS, C_TRAIN, "o", "DLBT — train",     cfg.N_MC, 0, 0),
        (dlbt["stim"],  cfg.TRAIN_TASKS, C_STIM,  "s", "DLBT — stim gen",  cfg.N_MC, 0, 1),
        (dlbt["task"],  cfg.VAL_TASKS,   C_TASK,  "^", "DLBT — task gen",  cfg.N_MC, 0, 2),
        (dlbt["joint"], cfg.VAL_TASKS,   C_JOINT, "D", "DLBT — joint gen", cfg.N_MC, 0, 3),
        (slda["train"], cfg.TRAIN_TASKS, C_TRAIN, "o", "SLDA — train",     None,     1, 0),
        (slda["stim"],  cfg.TRAIN_TASKS, C_STIM,  "s", "SLDA — stim gen",  None,     1, 1),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(14, 7),
                             sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.45, "wspace": 0.12})
    for ax in axes.flat:
        ax.set_visible(False)

    for pt, task_names, color, marker, title, mc_n, row, col in panels:
        ax = axes[row, col]
        ax.set_visible(True)
        _summary_scatter(ax, pt, task_names, color, marker, title, mc_n,
                         n_seeds=n_seeds, n_trials=n_trials)

    fig.supxlabel("Predicted P(right)", fontsize=12)
    fig.supylabel("True P(right)", fontsize=12, x=0.01)
    fig.suptitle(f"Behavioral prediction — {model_label}", fontsize=13)

    import seaborn as sns
    sns.despine(fig=fig, trim=False)
    plt.tight_layout(rect=[0.04, 0.04, 1, 1])
    out = plots_dir / f"plot_03_summary_{run_tag}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()

    # -----------------------------------------------------------------------
    # Plot 4 — Per-task DLBT grid
    # -----------------------------------------------------------------------
    ALL_TASKS = cfg.TRAIN_TASKS + cfg.VAL_TASKS
    N_COLS    = 8
    N_ROWS    = math.ceil(len(ALL_TASKS) / N_COLS)

    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(N_COLS * 2.0, N_ROWS * 2.2),
                             sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.55, "wspace": 0.08})
    for ax in axes.flat[len(ALL_TASKS):]:
        ax.set_visible(False)

    for ax, task_name in zip(axes.flat, ALL_TASKS):
        ax.plot([0, 1], [0, 1], ls=":", color="gray", lw=0.7, zorder=0)
        is_val = task_name in cfg.VAL_TASKS

        def _rho_mean(cond, tn):
            if tn not in dlbt[cond]:
                return float("nan")
            d  = dlbt[cond][tn]
            pm = d["pred"].mean(axis=0) if d["pred"].ndim == 2 else d["pred"]
            r, _ = spearmanr(pm, d["true"])
            return r

        if not is_val:
            for cond, color in [("train", C_TRAIN), ("stim", C_STIM)]:
                if task_name in dlbt[cond]:
                    d         = dlbt[cond][task_name]
                    pred_mean = d["pred"].mean(axis=0)
                    pred_sem  = d["pred"].std(axis=0) / np.sqrt(n_seeds)
                    true_vals = d["true"]
                    true_sem  = np.sqrt(true_vals * (1 - true_vals) / n_trials)
                    ax.errorbar(pred_mean, true_vals,
                                xerr=pred_sem, yerr=true_sem,
                                fmt="o", ms=3, alpha=0.2, color=color,
                                elinewidth=0.4, capsize=0, linewidth=0)
            ax.text(0.05, 0.93, f"ρ={_rho_mean('train', task_name):.2f}",
                    transform=ax.transAxes, fontsize=6, color=C_TRAIN, va="top")
            ax.text(0.05, 0.78, f"ρ={_rho_mean('stim', task_name):.2f}",
                    transform=ax.transAxes, fontsize=6, color=C_STIM, va="top")
        else:
            for cond, color in [("task", C_TASK), ("joint", C_JOINT)]:
                if task_name in dlbt[cond]:
                    d         = dlbt[cond][task_name]
                    pred_mean = d["pred"].mean(axis=0)
                    pred_sem  = d["pred"].std(axis=0) / np.sqrt(n_seeds)
                    true_vals = d["true"]
                    true_sem  = np.sqrt(true_vals * (1 - true_vals) / n_trials)
                    ax.errorbar(pred_mean, true_vals,
                                xerr=pred_sem, yerr=true_sem,
                                fmt="o", ms=3, alpha=0.2, color=color,
                                elinewidth=0.4, capsize=0, linewidth=0)
            ax.text(0.05, 0.93, f"ρ={_rho_mean('task', task_name):.2f}",
                    transform=ax.transAxes, fontsize=6, color=C_TASK, va="top")
            ax.text(0.05, 0.78, f"ρ={_rho_mean('joint', task_name):.2f}",
                    transform=ax.transAxes, fontsize=6, color=C_JOINT, va="top")

        ax.set_title(task_name.replace("_and_", " & "), fontsize=7, pad=3)
        ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.tick_params(labelsize=6)

    fig.supxlabel("Predicted P(right)", fontsize=10, y=0.01)
    fig.supylabel("True P(right)", fontsize=10, x=0.01)
    fig.suptitle(f"Per-task scatter — DLBT ({model_label})", fontsize=11)

    legend_handles = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor=C_TRAIN, ms=5, label="train"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor=C_STIM,  ms=5, label="stim gen"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor=C_TASK,  ms=5, label="task gen"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor=C_JOINT, ms=5, label="joint gen"),
    ]
    fig.legend(handles=legend_handles, loc="lower right", fontsize=8,
               bbox_to_anchor=(1.0, 0.02), ncol=2)

    plt.tight_layout(rect=[0.04, 0.04, 1, 1])
    out = plots_dir / f"plot_04_per_task_dlbt_{run_tag}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()

    # -----------------------------------------------------------------------
    # Plot 5 — Per-task SLDA grid
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(N_COLS * 2.0, N_ROWS * 2.2),
                             sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.55, "wspace": 0.08})
    for ax in axes.flat[len(ALL_TASKS):]:
        ax.set_visible(False)

    for ax, task_name in zip(axes.flat, ALL_TASKS):
        ax.plot([0, 1], [0, 1], ls=":", color="gray", lw=0.7, zorder=0)
        is_val = task_name in cfg.VAL_TASKS
        if is_val:
            ax.set_title(task_name.replace("_and_", " & "), fontsize=7, pad=3)
            ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([0, 0.5, 1])
            ax.tick_params(labelsize=6)
            continue

        for cond, color in [("train", C_TRAIN), ("stim", C_STIM)]:
            if task_name in slda[cond]:
                d = slda[cond][task_name]
                true_vals = d["true"]
                true_sem  = np.sqrt(true_vals * (1 - true_vals) / n_trials)
                ax.errorbar(d["pred"], true_vals,
                            yerr=true_sem,
                            fmt="s", ms=3, alpha=0.2, color=color,
                            elinewidth=0.4, capsize=0, linewidth=0)

        def _rho_slda(cond, tn):
            if tn not in slda[cond]:
                return float("nan")
            d = slda[cond][tn]
            r, _ = spearmanr(d["pred"], d["true"])
            return r

        ax.text(0.05, 0.93, f"ρ={_rho_slda('train', task_name):.2f}",
                transform=ax.transAxes, fontsize=6, color=C_TRAIN, va="top")
        ax.text(0.05, 0.78, f"ρ={_rho_slda('stim', task_name):.2f}",
                transform=ax.transAxes, fontsize=6, color=C_STIM, va="top")

        ax.set_title(task_name.replace("_and_", " & "), fontsize=7, pad=3)
        ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.tick_params(labelsize=6)

    fig.supxlabel("Predicted P(right)", fontsize=10, y=0.01)
    fig.supylabel("True P(right)", fontsize=10, x=0.01)
    fig.suptitle("Per-task scatter — SLDA", fontsize=11)

    legend_handles = [
        Line2D([0],[0], marker="s", color="w", markerfacecolor=C_TRAIN, ms=5, label="train"),
        Line2D([0],[0], marker="s", color="w", markerfacecolor=C_STIM,  ms=5, label="stim gen"),
    ]
    fig.legend(handles=legend_handles, loc="lower right", fontsize=8,
               bbox_to_anchor=(1.0, 0.02), ncol=2)

    plt.tight_layout(rect=[0.04, 0.04, 1, 1])
    out = plots_dir / f"plot_05_per_task_slda_{run_tag}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()

print("\nDone.")

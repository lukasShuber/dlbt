"""
01_fit/analysis.py — plots for the 01_fit training run.

Generated figures (per results pkl):
  plot_01_curves_<tag>.png        — learning curves: train + eval traces,
                                    + all-region cMSE traces, + noise floors
  plot_02_train_<tag>.png         — train region: pooled scatter + per-task grid
  plot_03_stim_gen_<tag>.png      — stim gen region: pooled + per-task grid
  plot_04_task_gen_<tag>.png      — task gen region: pooled + per-task grid
  plot_05_joint_gen_<tag>.png     — joint gen region: pooled + per-task grid
  plot_06_slda_<tag>.png          — SLDA per-task grid (train + stim_gen)

Run from repo root:
    python experiments/behavior/run0/01_fit/analysis.py
"""

import argparse
import json
import math
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

# ---------------------------------------------------------------------------
# Colours & helpers
# ---------------------------------------------------------------------------
C_TRAIN  = cfg.C_TRAIN
C_EVAL   = cfg.C_EVAL
C_STIM   = cfg.C_STIM
C_TASK   = cfg.C_TASK
C_JOINT  = cfg.C_JOINT

C_DLBT = "#C44F52"
C_SLDA = "#7D6EAE"

plots_dir = cfg.RESULTS_DIR / "plots"
plots_dir.mkdir(exist_ok=True)


def _true_sem(true_vals: np.ndarray, totals: np.ndarray) -> np.ndarray:
    totals_safe = np.clip(totals, 1, None)
    sem = np.sqrt(np.clip(true_vals * (1 - true_vals), 0, None) / totals_safe)
    sem[totals <= 0] = 0
    return sem


def _noise_floor_local(true_vals: np.ndarray, totals: np.ndarray) -> float:
    mask = totals > 1
    if not mask.any():
        return 0.0
    tv = true_vals[mask]
    return float(np.mean(tv * (1 - tv) / (totals[mask] - 1)))


def _compute_metrics(pt: dict, task_names: list, mc_n=None, n_seeds=1):
    """Aggregate predictions across tasks, return (pred_mean, pred_sem, true, totals, rho, mse, net_mse)."""
    all_preds  = np.concatenate([pt[t]["pred"]   for t in task_names if t in pt], axis=-1)
    all_trues  = np.concatenate([pt[t]["true"]   for t in task_names if t in pt])
    all_totals = np.concatenate([pt[t]["totals"] for t in task_names if t in pt])

    valid = all_totals > 0
    all_preds  = all_preds[..., valid] if all_preds.ndim == 2 else all_preds[valid]
    all_trues  = all_trues[valid]
    all_totals = all_totals[valid]

    if all_preds.ndim == 2:
        pred_mean = all_preds.mean(axis=0)
        pred_sem  = all_preds.std(axis=0) / np.sqrt(max(n_seeds, 1))
    else:
        pred_mean = all_preds
        pred_sem  = np.zeros_like(pred_mean)

    raw_mse = float(np.mean((pred_mean - all_trues) ** 2))
    if mc_n and mc_n > 1:
        raw_mse -= float(np.mean(pred_mean * (1 - pred_mean))) / (mc_n - 1)
    nf      = _noise_floor_local(all_trues, all_totals)
    net_mse = raw_mse - nf
    rho, _  = spearmanr(pred_mean, all_trues)
    return pred_mean, pred_sem, all_trues, all_totals, rho, raw_mse, net_mse


def _draw_pooled(ax, pred_mean, pred_sem, all_trues, all_totals, rho, raw_mse, net_mse,
                 color, marker="o", title="", n_seeds=1):
    true_sem = _true_sem(all_trues, all_totals)
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1.2, zorder=0)
    ax.errorbar(pred_mean, all_trues,
                xerr=pred_sem, yerr=true_sem,
                fmt=marker, ms=4, alpha=0.5, color=color,
                elinewidth=0.5, capsize=0, linewidth=0)
    ax.set_title(
        f"{title}\nMSE={raw_mse:.4f}  (−NF)={net_mse:+.4f}   ρ={rho:.3f}",
        fontsize=8, pad=4,
    )
    ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    ax.tick_params(labelsize=9)


def _draw_task_panel(ax, pt: dict, task_name: str, cond_colors: list, n_seeds=1, mc_n=None):
    """Draw a single per-task scatter panel."""
    ax.plot([0, 1], [0, 1], ls=":", color="gray", lw=0.7, zorder=0)
    y_top = 0.93
    for cond, color in cond_colors:
        if task_name not in pt[cond]:
            continue
        d     = pt[cond][task_name]
        p     = d["pred"]
        valid = d["totals"] > 0
        pv    = p[..., valid] if p.ndim == 2 else p[valid]
        pred_mean = pv.mean(axis=0) if pv.ndim == 2 else pv
        true_vals = d["true"][valid]
        totals    = d["totals"][valid]
        true_sem  = _true_sem(true_vals, totals)
        pred_sem  = pv.std(axis=0) / np.sqrt(max(n_seeds, 1)) if pv.ndim == 2 else np.zeros_like(pred_mean)
        ax.errorbar(pred_mean, true_vals,
                    xerr=pred_sem, yerr=true_sem,
                    fmt='o', ms=3, alpha=0.6, color=color,
                    elinewidth=0.4, capsize=0, linewidth=0)
        if valid.sum() >= 2:
            rho, _ = spearmanr(pred_mean, true_vals)
            raw_mse = float(np.mean((pred_mean - true_vals) ** 2))
            if mc_n and mc_n > 1:
                raw_mse -= float(np.mean(pred_mean * (1 - pred_mean))) / (mc_n - 1)
            nf      = _noise_floor_local(true_vals, totals)
            net_mse = raw_mse - nf
            ax.text(0.05, y_top, f"ρ={rho:.2f}",
                    transform=ax.transAxes, fontsize=6, color=color, va="top")
            y_top -= 0.13
            ax.text(0.05, y_top, f"mse={net_mse:.3f}",
                    transform=ax.transAxes, fontsize=6, color=color, va="top")
            y_top -= 0.13
    label = task_name.replace("_and_", " & ").replace("_", "/")
    ax.set_title(label, fontsize=7, pad=2)
    ax.set(xlim=(-0.05, 1.05), ylim=(-0.05, 1.05))
    ax.tick_params(labelsize=5)


def _region_figure(pt, region_name, task_list, cond_colors, color,
                   run_tag, n_seeds, mc_n, noise_floor_val):
    """
    Build one figure for a region: left = pooled scatter, right = per-task grid.
    Layout: gridspec with first column wider (pooled), then N_COLS task panels.
    """
    N_TASK_COLS = 6
    n_tasks     = len(task_list)
    n_task_rows = math.ceil(n_tasks / N_TASK_COLS)

    # Merge all preds across conds
    merged_pt: dict = {}
    for cond, _ in cond_colors:
        for t in task_list:
            if t in pt[cond]:
                if t not in merged_pt:
                    merged_pt[t] = pt[cond][t]

    pred_mean, pred_sem, all_trues, all_totals, rho, raw_mse, net_mse = \
        _compute_metrics(merged_pt, task_list, mc_n=mc_n, n_seeds=n_seeds)

    # Figure layout: pooled on the left (2 cols wide), per-task on the right
    total_cols   = 2 + N_TASK_COLS
    total_rows   = max(n_task_rows, 2)
    fig_w        = total_cols * 1.8 + 0.5
    fig_h        = total_rows * 2.0 + 0.6

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = gridspec.GridSpec(total_rows, total_cols,
                            hspace=0.6, wspace=0.25,
                            figure=fig)

    # Pooled scatter: spans all rows in first 2 columns
    ax_pooled = fig.add_subplot(gs[:total_rows, :2])
    _draw_pooled(ax_pooled, pred_mean, pred_sem, all_trues, all_totals,
                 rho, raw_mse, net_mse, color=color,
                 title=f"{region_name.replace('_', ' ').title()} — pooled",
                 n_seeds=n_seeds)
    ax_pooled.set_xlabel("Predicted P(yes)", fontsize=9)
    ax_pooled.set_ylabel("Human P(yes)",     fontsize=9)

    # Noise floor on pooled panel
    if noise_floor_val is not None:
        ax_pooled.axvline(x=-1, lw=0)  # dummy to make legend work
        ax_pooled.axhline(y=-1, lw=0)
        ax_pooled.text(0.97, 0.03,
                       f"NF={noise_floor_val:.4f}",
                       transform=ax_pooled.transAxes, fontsize=7,
                       ha="right", va="bottom", color="gray")

    # Per-task panels
    axes_flat = []
    for r in range(n_task_rows):
        for c in range(N_TASK_COLS):
            ax = fig.add_subplot(gs[r, 2 + c])
            axes_flat.append(ax)

    for ax in axes_flat[n_tasks:]:
        ax.set_visible(False)

    for ax, task_name in zip(axes_flat, task_list):
        _draw_task_panel(ax, pt, task_name, cond_colors, n_seeds=n_seeds, mc_n=mc_n)

    # Bottom row labels
    for i, ax in enumerate(axes_flat[:n_tasks]):
        row_i = i // N_TASK_COLS
        col_i = i  % N_TASK_COLS
        if row_i == n_task_rows - 1 or i >= n_tasks - N_TASK_COLS:
            ax.set_xlabel("Pred", fontsize=7)
        if col_i == 0:
            ax.set_ylabel("Human", fontsize=7)

    # Legend for per-task cond colours
    handles = [
        Line2D([0],[0], marker="o", color="w",
               markerfacecolor=c, markersize=5, label=cond)
        for cond, c in cond_colors
    ]
    fig.legend(handles=handles, loc="lower right",
               bbox_to_anchor=(1.0, 0.0),
               fontsize=8, frameon=False, ncol=len(cond_colors))

    sns.despine(fig=fig, trim=True)
    out = plots_dir / f"plot_{region_name}_{run_tag}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--tag", default=None)
args = parser.parse_args()

candidates = sorted(cfg.RESULTS_DIR.glob("results_*.pkl"))
if args.tag:
    candidates = [p for p in candidates if args.tag in p.stem]

if not candidates:
    raise FileNotFoundError(
        f"No results files found in {cfg.RESULTS_DIR}. Run run.py first."
    )

for results_path in candidates:
    run_tag = results_path.stem[len("results_"):]
    print(f"\n=== Processing: {results_path.name} (run_tag={run_tag}) ===")

    with open(results_path, "rb") as f:
        res = pickle.load(f)

    model_label    = res["model_label"]
    phase_boundary = res["phase_boundary"]
    best_epoch     = res["best_epoch"]
    noise_floors   = res.get("noise_floors", {})
    curves         = res["curves"]
    dlbt           = res["dlbt"]
    slda           = res["slda"]
    n_seeds        = res.get("n_seeds", 1)

    has_phase2 = phase_boundary < len(curves["train_nlls"]) - 1

    # -----------------------------------------------------------------------
    # Plot 01 — learning curves: 2 primary traces + all-region cMSE
    # -----------------------------------------------------------------------
    epochs = range(len(curves["train_nlls"]))
    has_stim  = "stim_mses"  in curves
    has_task  = "task_mses"  in curves
    has_joint = "joint_mses" in curves

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax_nll, ax_mse = axes

    # NLL: only train + eval (the 2 primary traces)
    ax_nll.plot(epochs, curves["train_nlls"], color=C_TRAIN, label="train",    lw=1.2)
    ax_nll.plot(epochs, curves["eval_nlls"],  color=C_EVAL,  label="eval",     lw=1.2)
    ax_nll.axvline(best_epoch, ls=":", color="gray", lw=0.8)
    if has_phase2:
        ax_nll.axvline(phase_boundary, ls="--", color="black", lw=0.8, alpha=0.5)
    ax_nll.set(ylabel="NLL", xlabel="epoch",
               title=f"{model_label} — NLL (train / eval)")
    ax_nll.legend(fontsize=8)
    ax_nll.set_ylim(bottom=0)

    # cMSE: train, eval, stim, task, joint
    ax_mse.plot(epochs, curves["train_mses"], color=C_TRAIN,  label="train",     lw=1.2)
    ax_mse.plot(epochs, curves["eval_mses"],  color=C_EVAL,   label="eval",      lw=1.2)
    if has_stim:
        ax_mse.plot(epochs, curves["stim_mses"],  color=C_STIM,   label="stim gen",  lw=1.0, alpha=0.7)
    if has_task:
        ax_mse.plot(epochs, curves["task_mses"],  color=C_TASK,   label="task gen",  lw=1.0, alpha=0.7)
    if has_joint:
        ax_mse.plot(epochs, curves["joint_mses"], color=C_JOINT,  label="joint gen", lw=1.0, alpha=0.7)
    ax_mse.axvline(best_epoch, ls=":", color="gray", lw=0.8)
    if has_phase2:
        ax_mse.axvline(phase_boundary, ls="--", color="black", lw=0.8, alpha=0.5)

    # Noise-floor reference lines
    for key, color in [("train", C_TRAIN), ("eval", C_EVAL),
                       ("stim_gen", C_STIM), ("task_gen", C_TASK),
                       ("joint_gen", C_JOINT)]:
        if key in noise_floors:
            ax_mse.axhline(noise_floors[key], ls="--", color=color, alpha=0.35, lw=1)

    ax_mse.set(ylabel="cMSE", xlabel="epoch",
               title=f"{model_label} — cMSE")
    ax_mse.legend(fontsize=8)
    ax_mse.set_ylim(bottom=0)

    sns.despine(trim=True)
    plt.tight_layout()
    out = plots_dir / f"plot_01_curves_{run_tag}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()

    # -----------------------------------------------------------------------
    # Plot 02–05 — one figure per region
    # -----------------------------------------------------------------------
    # (train and eval share the same image pool — main images, train tasks)
    # For the per-region figure we show predictions from both conditions
    # (train + eval) in the pooled scatter, and per-task panels coloured
    # by condition.

    # train region
    _region_figure(
        pt         = dlbt,
        region_name= "02_train",
        task_list  = cfg.TRAIN_TASKS,
        cond_colors= [("train", C_TRAIN)],
        color      = C_TRAIN,
        run_tag    = run_tag,
        n_seeds    = n_seeds,
        mc_n       = cfg.N_MC,
        noise_floor_val = noise_floors.get("train"),
    )

    # eval region (in-dist held-out cells, same image pool as train)
    _region_figure(
        pt         = dlbt,
        region_name= "02b_eval",
        task_list  = cfg.TRAIN_TASKS,
        cond_colors= [("eval", C_EVAL)],
        color      = C_EVAL,
        run_tag    = run_tag,
        n_seeds    = n_seeds,
        mc_n       = cfg.N_MC,
        noise_floor_val = noise_floors.get("eval"),
    )

    # stim gen (probe images, train tasks)
    _region_figure(
        pt         = dlbt,
        region_name= "03_stim_gen",
        task_list  = cfg.TRAIN_TASKS,
        cond_colors= [("stim", C_STIM)],
        color      = C_STIM,
        run_tag    = run_tag,
        n_seeds    = n_seeds,
        mc_n       = cfg.N_MC,
        noise_floor_val = noise_floors.get("stim_gen"),
    )

    # task gen (main images, val tasks)
    _region_figure(
        pt         = dlbt,
        region_name= "04_task_gen",
        task_list  = cfg.VAL_TASKS,
        cond_colors= [("task", C_TASK)],
        color      = C_TASK,
        run_tag    = run_tag,
        n_seeds    = n_seeds,
        mc_n       = cfg.N_MC,
        noise_floor_val = noise_floors.get("task_gen"),
    )

    # joint gen (probe images, val tasks)
    _region_figure(
        pt         = dlbt,
        region_name= "05_joint_gen",
        task_list  = cfg.VAL_TASKS,
        cond_colors= [("joint", C_JOINT)],
        color      = C_JOINT,
        run_tag    = run_tag,
        n_seeds    = n_seeds,
        mc_n       = cfg.N_MC,
        noise_floor_val = noise_floors.get("joint_gen"),
    )

    # -----------------------------------------------------------------------
    # Plot 06 — SLDA per-task grid (train + stim_gen) — skipped for Flex runs
    # -----------------------------------------------------------------------
    if not slda or not any(slda.get(c) for c in ("train", "stim")):
        print("Skipping SLDA plot (no SLDA predictions in this results file).")
        continue

    N_COLS_S = 6
    N_ROWS_S = math.ceil(len(cfg.TRAIN_TASKS) / N_COLS_S)
    fig, axes = plt.subplots(N_ROWS_S, N_COLS_S,
                             figsize=(N_COLS_S * 2.0, N_ROWS_S * 2.2),
                             sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.55, "wspace": 0.08})
    for ax in axes.flat[len(cfg.TRAIN_TASKS):]:
        ax.set_visible(False)

    for idx, (ax, task_name) in enumerate(zip(axes.flat, cfg.TRAIN_TASKS)):
        ax.plot([0, 1], [0, 1], ls=":", color="gray", lw=0.7, zorder=0)
        y_top = 0.93
        for cond, color in [("train", C_TRAIN), ("stim", C_STIM)]:
            if task_name not in slda[cond]:
                continue
            d     = slda[cond][task_name]
            valid = d["totals"] > 0
            true_sem = _true_sem(d["true"][valid], d["totals"][valid])
            ax.errorbar(d["pred"][valid], d["true"][valid],
                        yerr=true_sem,
                        fmt="s", ms=3, alpha=0.5, color=color,
                        elinewidth=0.4, capsize=0, linewidth=0)
            if valid.sum() >= 2:
                r, _ = spearmanr(d["pred"][valid], d["true"][valid])
                ax.text(0.05, y_top, f"ρ={r:.2f}",
                        transform=ax.transAxes, fontsize=6, color=color, va="top")
                y_top -= 0.15
        ax.set_title(task_name.replace("_and_", " & ").replace("_", "/"),
                     fontsize=7, pad=2)
        row_i, col_i = divmod(idx, N_COLS_S)
        if row_i == N_ROWS_S - 1:
            ax.set_xlabel("Pred",  fontsize=7)
        if col_i == 0:
            ax.set_ylabel("Human", fontsize=7)
        ax.set(xlim=(-0.05, 1.05), ylim=(-0.05, 1.05))
        ax.tick_params(labelsize=5)

    fig.legend(handles=[
        Line2D([0],[0], marker="s", color="w", markerfacecolor=c, markersize=5, label=l)
        for c, l in [(C_TRAIN,"train"),(C_STIM,"stim gen")]
    ], loc="lower right", bbox_to_anchor=(1.0, 0.0), fontsize=7, frameon=False)
    fig.text(0.5, -0.01, "Predicted P(yes)", ha="center", fontsize=9)
    fig.text(-0.01, 0.5, "Human P(yes)", va="center", rotation="vertical", fontsize=9)
    sns.despine(fig=fig, trim=True)
    out = plots_dir / f"plot_06_slda_{run_tag}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()

print("\nAll plots saved to", plots_dir)

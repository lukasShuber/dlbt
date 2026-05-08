"""
run1/02_data_efficiency/analysis.py — plots for the data-efficiency sweep.

Figures produced per results pkl:
  plot_01_cmse_vs_budget_<tag>.png              — cMSE−NF vs trial budget
  plot_02_curves_<tag>_budget<B>.png            — learning curves per budget
  plot_03_stim_gen_scatter_<tag>_budget<B>.png  — stim gen pooled scatter per budget
  plot_03_stim_gen_grid_<tag>_budget<B>.png     — stim gen per-task grid per budget
  plot_04_joint_gen_scatter_<tag>_budget<B>.png — joint gen pooled scatter per budget
  plot_04_joint_gen_grid_<tag>_budget<B>.png    — joint gen per-task grid per budget

Stim gen is always plotted (relevant for SPLIT_MODE="all").
Joint gen is plotted only when val_tasks is non-empty.

Run from repo root:
    python experiments/behavior/run1/02_data_efficiency/analysis.py [--tag TAG]
"""

import argparse
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ARITY_COLOR = {1: "#2a6fb5", 2: "#43AA8B", 3: "#E76F51", 4: "#9B5DE5"}
N_TASK_COLS = 8

plots_dir = cfg.RESULTS_DIR / "plots"
plots_dir.mkdir(exist_ok=True)


def _arity(t: str) -> int:
    return t.count("_and_") + 1


def _label(t: str) -> str:
    return t.replace("_and_", " & ").replace("_", "/")


def _true_sem(true_vals: np.ndarray, totals: np.ndarray) -> np.ndarray:
    safe = np.clip(totals, 1, None)
    sem  = np.sqrt(np.clip(true_vals * (1 - true_vals), 0, None) / safe)
    sem[totals <= 0] = 0
    return sem


def _noise_floor_local(true_vals: np.ndarray, totals: np.ndarray) -> float:
    mask = totals > 1
    if not mask.any():
        return 0.0
    return float(np.mean(true_vals[mask] * (1 - true_vals[mask]) / (totals[mask] - 1)))


def _compute_metrics(region_preds: dict, task_names: list, mc_n: int):
    """Pool predictions across tasks. Returns None if no data."""
    present = [t for t in task_names if t in region_preds]
    if not present:
        return None
    pred   = np.concatenate([region_preds[t]["pred"]   for t in present])
    true   = np.concatenate([region_preds[t]["true"]   for t in present])
    totals = np.concatenate([region_preds[t]["totals"] for t in present])
    valid  = totals > 0
    pred, true, totals = pred[valid], true[valid], totals[valid]
    if len(pred) < 2:
        return None
    raw_mse = float(np.mean((pred - true) ** 2))
    if mc_n > 1:
        raw_mse -= float(np.mean(pred * (1 - pred))) / (mc_n - 1)
    nf      = _noise_floor_local(true, totals)
    net_mse = raw_mse - nf
    rho, _  = spearmanr(pred, true)
    return pred, true, totals, rho, raw_mse, net_mse


def _plot_summary(region_preds, task_names, mc_n, color, title,
                  noise_floor_val, run_tag, fname):
    """Standalone pooled scatter figure."""
    metrics = _compute_metrics(region_preds, task_names, mc_n)
    if metrics is None:
        print(f"  Skipping summary {fname}: no data.")
        return
    pred, true, totals, rho, raw_mse, net_mse = metrics
    ts = _true_sem(true, totals)

    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1.2, zorder=0)
    ax.errorbar(pred, true, yerr=ts,
                fmt="o", ms=4, alpha=0.45, color=color,
                elinewidth=0.5, capsize=0, linewidth=0)
    ax.set_title(
        f"{title}\nMSE={raw_mse:.4f}  (−NF)={net_mse:+.4f}   ρ={rho:.3f}",
        fontsize=8, pad=4,
    )
    ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("Predicted P(yes)", fontsize=9)
    ax.set_ylabel("Human P(yes)",     fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    if noise_floor_val is not None:
        ax.text(0.97, 0.03, f"NF={noise_floor_val:.4f}",
                transform=ax.transAxes, fontsize=7,
                ha="right", va="bottom", color="gray")
    sns.despine(fig=fig, trim=True)
    plt.tight_layout()
    out = plots_dir / fname
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()


def _plot_task_grid(region_preds, task_names, mc_n, run_tag, fname):
    """Standalone per-task grid figure."""
    task_names = sorted(task_names, key=lambda t: (_arity(t), t))
    present    = [t for t in task_names if t in region_preds]
    if not present:
        print(f"  Skipping grid {fname}: no data.")
        return

    n_tasks     = len(task_names)
    n_task_rows = math.ceil(n_tasks / N_TASK_COLS)

    fig, axes = plt.subplots(
        n_task_rows, N_TASK_COLS,
        figsize=(N_TASK_COLS * 1.8, n_task_rows * 2.0),
        gridspec_kw={"hspace": 0.65, "wspace": 0.10},
    )
    axes_flat = np.atleast_2d(axes).flatten()
    for ax in axes_flat[n_tasks:]:
        ax.set_visible(False)

    for i, (ax, task_name) in enumerate(zip(axes_flat, task_names)):
        ax.plot([0, 1], [0, 1], ls=":", color="gray", lw=0.7, zorder=0)
        color = ARITY_COLOR.get(_arity(task_name), "#555")

        if task_name not in region_preds:
            ax.set_visible(False)
            continue

        d     = region_preds[task_name]
        valid = d["totals"] > 0
        pm    = d["pred"][valid]
        tv    = d["true"][valid]
        tot   = d["totals"][valid]
        ts    = _true_sem(tv, tot)

        ax.errorbar(pm, tv, yerr=ts,
                    fmt="o", ms=3, alpha=0.7, color=color,
                    elinewidth=0.4, capsize=0, linewidth=0)

        y_top = 0.95
        if valid.sum() >= 2:
            rho, _ = spearmanr(pm, tv)
            raw    = float(np.mean((pm - tv) ** 2))
            if mc_n > 1:
                raw -= float(np.mean(pm * (1 - pm))) / (mc_n - 1)
            net = raw - _noise_floor_local(tv, tot)
            ax.text(0.05, y_top,        f"ρ={rho:.2f}", transform=ax.transAxes,
                    fontsize=5.5, color=color, va="top")
            ax.text(0.05, y_top - 0.15, f"m={net:.3f}", transform=ax.transAxes,
                    fontsize=5.5, color=color, va="top")

        ax.set_title(_label(task_name), fontsize=6, pad=2, color=color)
        ax.set(xlim=(-0.05, 1.05), ylim=(-0.05, 1.05))
        ax.tick_params(labelsize=4.5)
        row_i, col_i = divmod(i, N_TASK_COLS)
        if row_i == n_task_rows - 1 or i >= n_tasks - N_TASK_COLS:
            ax.set_xlabel("Pred",  fontsize=6)
        if col_i == 0:
            ax.set_ylabel("Human", fontsize=6)

    handles = [Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=c, markersize=5, label=f"{a}-way")
               for a, c in ARITY_COLOR.items()]
    fig.legend(handles=handles, loc="lower right",
               bbox_to_anchor=(1.0, 0.0), fontsize=7,
               frameon=False, ncol=4)
    sns.despine(fig=fig, trim=True)
    out = plots_dir / fname
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--tag", default=None,
                    help="Filter results pkl by tag substring (default: use cfg.RUN_TAG).")
args = parser.parse_args()

candidates = sorted(cfg.RESULTS_DIR.glob("data_efficiency_*.pkl"))
if args.tag:
    candidates = [p for p in candidates if args.tag in p.stem]
else:
    candidates = [p for p in candidates if cfg.RUN_TAG in p.stem]

if not candidates:
    raise FileNotFoundError(
        f"No data_efficiency_*.pkl found in {cfg.RESULTS_DIR}. Run run.py first."
    )

for results_path in candidates:
    run_tag = results_path.stem[len("data_efficiency_"):]
    print(f"\n=== {results_path.name}  (run_tag={run_tag}) ===")

    with open(results_path, "rb") as f:
        summary = pickle.load(f)

    results   = summary["results"]
    nfs       = summary["noise_floors"]
    n_pool    = summary["n_pool"]
    has_corr  = summary.get("threshold_correction", False)
    train_tasks = summary.get("train_tasks", cfg.TRAIN_TASKS)
    val_tasks   = sorted(
        summary.get("val_tasks", cfg.VAL_TASKS),
        key=lambda t: (_arity(t), t),
    )
    train_tasks_sorted = sorted(train_tasks, key=lambda t: (_arity(t), t))

    # -------------------------------------------------------------------
    # Sort budgets by trial count
    # -------------------------------------------------------------------
    all_points = []
    for label, res in results.items():
        x = n_pool if label == "full" else int(label)
        all_points.append((x, label, res))
    all_points.sort(key=lambda p: p[0])

    x_all    = [p[0] for p in all_points]
    lab_all  = [p[1] for p in all_points]
    full_idx = next((i for i, p in enumerate(all_points) if p[1] == "full"),
                    len(all_points) - 1)

    # -------------------------------------------------------------------
    # Plot 01 — cMSE−NF vs trial budget
    #
    # Traces: SLDA stim gen | DLBT stim gen | DLBT joint gen
    # Reference: random guesser (P=0.5), computed once from full-budget preds.
    # Values pulled directly from stored per-budget metrics (avoids NF mismatch).
    # -------------------------------------------------------------------
    C_DLBT = "#C44F52"
    C_SLDA = "#7D6EAE"
    C_RAND = "#999999"

    x_plot  = x_all[:full_idx + 1]
    lab_plot = lab_all[:full_idx + 1]

    # Random guesser cMSE-NF — constant across budgets, compute once
    _full_preds = all_points[full_idx][2].get("preds", {})
    _stim_p = _full_preds.get("stim_gen", {})
    if _stim_p:
        _true   = np.concatenate([_stim_p[t]["true"]   for t in _stim_p])
        _totals = np.concatenate([_stim_p[t]["totals"] for t in _stim_p])
        _valid  = _totals > 0
        _nf     = nfs.get("stim_gen", 0.0)
        random_stim_cmse = float(np.mean((0.5 - _true[_valid]) ** 2)) - _nf
    else:
        random_stim_cmse = float("nan")

    fig, ax = plt.subplots(figsize=(5, 5))

    # Random guesser — horizontal reference
    if not np.isnan(random_stim_cmse):
        ax.axhline(random_stim_cmse, color=C_RAND, lw=1.5,
                   ls=(0, (4, 3)), label="Random (P=0.5)", zorder=1)

    # SLDA stim gen
    slda_y = [p[2].get("slda_stim_gen_cmse_net", float("nan"))
              for p in all_points[:full_idx + 1]]
    if not all(np.isnan(slda_y)):
        ax.plot(x_plot, slda_y, "o:", color=C_SLDA, lw=2.2, ms=6,
                label="SLDA — stim gen", zorder=3)

    # DLBT stim gen
    dlbt_stim_y = [p[2].get("stim_gen_cmse_net", float("nan"))
                   for p in all_points[:full_idx + 1]]
    ax.plot(x_plot, dlbt_stim_y, "o:", color=C_DLBT, lw=2.2, ms=6,
            label="DLBT — stim gen", zorder=4)

    # DLBT joint gen (only when val tasks exist)
    if val_tasks:
        dlbt_joint_y = [p[2].get("joint_gen_cmse_net", float("nan"))
                        for p in all_points[:full_idx + 1]]
        ax.plot(x_plot, dlbt_joint_y, "o-", color=C_DLBT, lw=2.2, ms=6,
                label="DLBT — joint gen", zorder=4)

    ax.set_xscale("log")
    ax.set_xlabel("Trial budget", fontsize=11)
    ax.set_ylabel("cMSE − noise floor", fontsize=11)
    ax.set_title(f"Data efficiency  [{run_tag}]", fontsize=10)
    ax.set_xticks(x_plot)
    ax.set_xticklabels(lab_plot, fontsize=9)
    ax.legend(fontsize=9, frameon=False)
    ax.set_ylim(bottom=0)
    sns.despine(trim=True)
    plt.tight_layout()
    out = plots_dir / f"plot_01_cmse_vs_budget_{run_tag}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()

    # -------------------------------------------------------------------
    # Plot 02 — learning curves per budget
    # -------------------------------------------------------------------
    for x, label, res in all_points:
        curves = res.get("curves")
        if not curves:
            continue
        epochs = range(len(curves["train_mses"]))

        fig, ax = plt.subplots(figsize=(8, 3.8))
        ax.plot(epochs, curves["train_mses"], color=cfg.C_TRAIN, label="train",    lw=1.2)
        ax.plot(epochs, curves["eval_mses"],  color=cfg.C_EVAL,  label="eval",     lw=1.2)
        for key, color, lbl in [
            ("stim_mses",  cfg.C_STIM,  "stim gen"),
            ("task_mses",  cfg.C_TASK,  "task gen"),
            ("joint_mses", cfg.C_JOINT, "joint gen"),
        ]:
            if curves.get(key):
                ax.plot(epochs, curves[key], color=color, label=lbl, lw=1.0, alpha=0.7)

        ax.axvline(res["best_epoch"], ls=":", color="gray", lw=0.8)
        for nf_key, color in [("eval",      cfg.C_EVAL),
                               ("stim_gen",  cfg.C_STIM),
                               ("task_gen",  cfg.C_TASK),
                               ("joint_gen", cfg.C_JOINT)]:
            if nf_key in nfs:
                ax.axhline(nfs[nf_key], ls="--", color=color, alpha=0.35, lw=1)

        ax.set(xlabel="epoch", ylabel="cMSE",
               title=f"Budget = {label}  "
                     f"(trials={res['n_trials']}, cells={res['n_cells']})")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8, frameon=False)
        sns.despine(trim=True)
        plt.tight_layout()
        out = plots_dir / f"plot_02_curves_{run_tag}_budget{label}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out}")
        plt.close()

    # -------------------------------------------------------------------
    # Plots 03 & 04 — per-budget scatter: stim gen + joint gen
    # -------------------------------------------------------------------
    for x, budget_label, res in all_points:
        preds_b = res.get("preds", {})

        # -- stim gen (always) --
        stim_preds = preds_b.get("stim_gen", {})
        _plot_summary(
            stim_preds, train_tasks_sorted, cfg.N_MC,
            color=cfg.C_STIM,
            title=f"Stim Gen — budget={budget_label}",
            noise_floor_val=nfs.get("stim_gen"),
            run_tag=run_tag,
            fname=f"plot_03_stim_gen_scatter_{run_tag}_budget{budget_label}.png",
        )
        _plot_task_grid(
            stim_preds, train_tasks_sorted, cfg.N_MC,
            run_tag=run_tag,
            fname=f"plot_03_stim_gen_grid_{run_tag}_budget{budget_label}.png",
        )

        # -- joint gen (only when val tasks exist) --
        if not val_tasks:
            continue
        joint_preds = preds_b.get("joint_gen", {})
        _plot_summary(
            joint_preds, val_tasks, cfg.N_MC,
            color=cfg.C_JOINT,
            title=f"Joint Gen — budget={budget_label}",
            noise_floor_val=nfs.get("joint_gen"),
            run_tag=run_tag,
            fname=f"plot_04_joint_gen_scatter_{run_tag}_budget{budget_label}.png",
        )
        _plot_task_grid(
            joint_preds, val_tasks, cfg.N_MC,
            run_tag=run_tag,
            fname=f"plot_04_joint_gen_grid_{run_tag}_budget{budget_label}.png",
        )

print(f"\nAll plots saved to {plots_dir}")

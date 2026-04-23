"""
01_fit/arity_correction.py
--------------------------
Tests the logit-space arity correction for the systematic under-prediction
of higher-arity tasks in joint generalisation.

The geometric bias of the simplex is additive in logit space:
  logit(p_corr) = logit(p_orig) + (arity - 1) * log(2)

Equivalently:
  p_corr = p_orig * 2^(n-1) / (p_orig * 2^(n-1) + (1 - p_orig))

Unlike simple multiplication this gives a larger absolute boost to small
probabilities and is naturally bounded in [0, 1] without clipping.

Produces one figure:
  plot_07_arity_correction_<tag>.png
    Left:  pooled scatter (original vs corrected, overlay)
    Right: per-task panels showing original (faded) + corrected (solid)
           with ρ and cMSE-NF for both.

Run from repo root:
    python experiments/behavior/run0/01_fit/arity_correction.py
"""

import argparse
import math
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

# ---------------------------------------------------------------------------
# Helpers (mirrors analysis.py)
# ---------------------------------------------------------------------------

plots_dir = cfg.RESULTS_DIR / "plots"
plots_dir.mkdir(exist_ok=True)

C_ORIG = cfg.C_JOINT          # original  — teal
C_CORR = "#E76F51"            # corrected — orange-red


def _noise_floor_local(true_vals: np.ndarray, totals: np.ndarray) -> float:
    mask = totals > 1
    if not mask.any():
        return 0.0
    return float(np.mean(true_vals[mask] * (1 - true_vals[mask]) / (totals[mask] - 1)))


def _true_sem(true_vals, totals):
    totals_safe = np.clip(totals, 1, None)
    sem = np.sqrt(np.clip(true_vals * (1 - true_vals), 0, None) / totals_safe)
    sem[totals <= 0] = 0
    return sem


def _arity(task_name: str) -> int:
    return task_name.count("_and_") + 1


def _correct(pred: np.ndarray, task_name: str) -> np.ndarray:
    """Logit-space arity correction.

    logit(p_corr) = logit(p_orig) + (n - 1) * log(2)

    Equivalent to:
        p_corr = p * k / (p * k + (1 - p)),   k = 2^(n-1)

    Unlike simple multiplication this gives a larger absolute boost to small
    probabilities and is naturally bounded in [0, 1].
    """
    n     = _arity(task_name)
    shift = (n - 1) * np.log(2)           # additive shift in logit space
    p     = np.clip(pred, 1e-7, 1 - 1e-7)
    return 1.0 / (1.0 + np.exp(-(np.log(p / (1 - p)) + shift)))


def _cmse_nf(pred_mean, true_vals, totals, mc_n):
    raw = float(np.mean((pred_mean - true_vals) ** 2))
    if mc_n and mc_n > 1:
        raw -= float(np.mean(pred_mean * (1 - pred_mean))) / (mc_n - 1)
    nf = _noise_floor_local(true_vals, totals)
    return raw - nf, nf


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
    raise FileNotFoundError(f"No results files found in {cfg.RESULTS_DIR}. Run run.py first.")

for results_path in candidates:
    run_tag = results_path.stem[len("results_"):]
    print(f"\n=== {results_path.name}  (run_tag={run_tag}) ===")

    with open(results_path, "rb") as f:
        res = pickle.load(f)

    dlbt   = res["dlbt"]
    mc_n   = cfg.N_MC
    n_seeds = res.get("n_seeds", 1)

    joint_pt = dlbt.get("joint", {})
    val_tasks = [t for t in cfg.VAL_TASKS if t in joint_pt]

    if not val_tasks:
        print("  No joint predictions found — skipping.")
        continue

    # -----------------------------------------------------------------------
    # Gather pooled arrays (original + corrected)
    # -----------------------------------------------------------------------
    orig_preds_all, corr_preds_all, trues_all, totals_all = [], [], [], []

    for t in val_tasks:
        d      = joint_pt[t]
        pred   = d["pred"]
        true   = d["true"]
        totals = d["totals"]
        valid  = totals > 0

        p   = pred[..., valid] if pred.ndim == 2 else pred[valid]
        pm  = p.mean(axis=0) if p.ndim == 2 else p
        tv  = true[valid]
        tot = totals[valid]

        pc = _correct(pm, t)

        orig_preds_all.append(pm)
        corr_preds_all.append(pc)
        trues_all.append(tv)
        totals_all.append(tot)

    orig_pool  = np.concatenate(orig_preds_all)
    corr_pool  = np.concatenate(corr_preds_all)
    trues_pool = np.concatenate(trues_all)
    tots_pool  = np.concatenate(totals_all)

    rho_orig, _ = spearmanr(orig_pool, trues_pool)
    rho_corr, _ = spearmanr(corr_pool, trues_pool)
    mse_orig, nf = _cmse_nf(orig_pool, trues_pool, tots_pool, mc_n)
    mse_corr, _  = _cmse_nf(corr_pool, trues_pool, tots_pool, mc_n)

    print(f"  Pooled ORIGINAL:   ρ={rho_orig:.3f}  cMSE-NF={mse_orig:+.4f}")
    print(f"  Pooled CORRECTED:  ρ={rho_corr:.3f}  cMSE-NF={mse_corr:+.4f}  (NF={nf:.4f})")

    # -----------------------------------------------------------------------
    # Figure layout
    # -----------------------------------------------------------------------
    N_TASK_COLS = 6
    n_tasks     = len(val_tasks)
    n_task_rows = math.ceil(n_tasks / N_TASK_COLS)

    total_cols = 2 + N_TASK_COLS
    total_rows = max(n_task_rows, 2)
    fig_w      = total_cols * 1.8 + 0.5
    fig_h      = total_rows * 2.0 + 0.6

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = gridspec.GridSpec(total_rows, total_cols,
                            hspace=0.6, wspace=0.25, figure=fig)

    # -----------------------------------------------------------------------
    # Left: pooled scatter
    # -----------------------------------------------------------------------
    ax_p = fig.add_subplot(gs[:total_rows, :2])
    ax_p.plot([0, 1], [0, 1], ls="--", color="gray", lw=1.2, zorder=0)
    ts_pool = _true_sem(trues_pool, tots_pool)
    ax_p.errorbar(orig_pool, trues_pool, yerr=ts_pool,
                  fmt="o", ms=4, alpha=0.35, color=C_ORIG,
                  elinewidth=0.4, capsize=0, linewidth=0, label="original")
    ax_p.errorbar(corr_pool, trues_pool, yerr=ts_pool,
                  fmt="o", ms=4, alpha=0.55, color=C_CORR,
                  elinewidth=0.4, capsize=0, linewidth=0, label="corrected")
    ax_p.set_title(
        f"Joint gen — pooled\n"
        f"orig:  ρ={rho_orig:.3f}  cMSE-NF={mse_orig:+.4f}\n"
        f"corr:  ρ={rho_corr:.3f}  cMSE-NF={mse_corr:+.4f}",
        fontsize=8, pad=4,
    )
    ax_p.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    ax_p.set_xticks([0, 0.5, 1]); ax_p.set_yticks([0, 0.5, 1])
    ax_p.set_xlabel("Predicted P(yes)", fontsize=9)
    ax_p.set_ylabel("Human P(yes)",     fontsize=9)
    ax_p.text(0.97, 0.03, f"NF={nf:.4f}",
              transform=ax_p.transAxes, fontsize=7, ha="right", va="bottom", color="gray")
    ax_p.legend(fontsize=7, loc="upper left", frameon=False)

    # -----------------------------------------------------------------------
    # Right: per-task panels
    # -----------------------------------------------------------------------
    axes_flat = []
    for r in range(n_task_rows):
        for c in range(N_TASK_COLS):
            axes_flat.append(fig.add_subplot(gs[r, 2 + c]))

    for ax in axes_flat[n_tasks:]:
        ax.set_visible(False)

    for ax, task_name in zip(axes_flat, val_tasks):
        d      = joint_pt[task_name]
        pred   = d["pred"]
        true   = d["true"]
        totals = d["totals"]
        valid  = totals > 0

        p   = pred[..., valid] if pred.ndim == 2 else pred[valid]
        pm  = p.mean(axis=0) if p.ndim == 2 else p
        tv  = true[valid]
        tot = totals[valid]
        pc  = _correct(pm, task_name)
        ts  = _true_sem(tv, tot)

        ax.plot([0, 1], [0, 1], ls=":", color="gray", lw=0.7, zorder=0)

        # original (faded)
        ax.errorbar(pm, tv, yerr=ts,
                    fmt="o", ms=3, alpha=0.3, color=C_ORIG,
                    elinewidth=0.3, capsize=0, linewidth=0)
        # corrected (solid)
        ax.errorbar(pc, tv, yerr=ts,
                    fmt="o", ms=3, alpha=0.7, color=C_CORR,
                    elinewidth=0.3, capsize=0, linewidth=0)

        y_top = 0.97
        n_way = _arity(task_name)
        shift = (n_way - 1) * np.log(2)
        ax.text(0.05, y_top, f"+{shift:.2f} logit  ({n_way}-way)",
                transform=ax.transAxes, fontsize=5.5, color="gray", va="top")
        y_top -= 0.14

        if valid.sum() >= 2:
            ro, _ = spearmanr(pm, tv)
            rc, _ = spearmanr(pc, tv)
            mo, _ = _cmse_nf(pm, tv, tot, mc_n)
            mc_v, _ = _cmse_nf(pc, tv, tot, mc_n)
            ax.text(0.05, y_top, f"ρ  {ro:.2f}→{rc:.2f}",
                    transform=ax.transAxes, fontsize=5.5, color=C_CORR, va="top")
            y_top -= 0.13
            ax.text(0.05, y_top, f"mse {mo:.3f}→{mc_v:.3f}",
                    transform=ax.transAxes, fontsize=5.5, color=C_CORR, va="top")

        label = task_name.replace("_and_", " & ").replace("_", "/")
        ax.set_title(label, fontsize=7, pad=2)
        ax.set(xlim=(-0.05, 1.05), ylim=(-0.05, 1.05))
        ax.tick_params(labelsize=5)

    for i, ax in enumerate(axes_flat[:n_tasks]):
        if i // N_TASK_COLS == n_task_rows - 1 or i >= n_tasks - N_TASK_COLS:
            ax.set_xlabel("Pred", fontsize=7)
        if i % N_TASK_COLS == 0:
            ax.set_ylabel("Human", fontsize=7)

    fig.legend(handles=[
        Line2D([0],[0], marker="o", color="w", markerfacecolor=C_ORIG,
               markersize=5, label="original", alpha=0.4),
        Line2D([0],[0], marker="o", color="w", markerfacecolor=C_CORR,
               markersize=5, label="corrected (logit + (n−1)·log2)"),
    ], loc="lower right", bbox_to_anchor=(1.0, 0.0), fontsize=8, frameon=False)

    sns.despine(fig=fig, trim=True)
    out = plots_dir / f"plot_07_arity_correction_{run_tag}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()

print("\nDone.")

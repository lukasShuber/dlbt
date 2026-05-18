"""
run1/023_efficiency_main/analysis.py — budget-sweep plots.

Produces two figures (cMSE−NF and Spearman ρ vs. trial budget):

  Traces (mean ± SEM across seeds):
    • DLBT          — red solid
    • SLDA          — purple solid
    • Anti-human    — gray solid  (cMSE plot only; omitted from ρ plot)

  Reference lines (no SEM):
    • Random guesser       — gray dashed horizontal   (cMSE only)
    • Random-init DLBT     — gray dotted horizontal   (cMSE only)

  Markers:
    • Budget grid points   — open markers, connected by lines
    • All-data point       — filled marker, plotted separately (not connected)

Run from repo root:
    python experiments/behavior/run1/023_efficiency_main/analysis.py
"""

import argparse
import pickle
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import NullFormatter, LogLocator

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--pkl", default=None,
                    help="Path to efficiency_main.pkl. Default: auto-discover.")
parser.add_argument("--log-y", action="store_true",
                    help="Log-scale y-axis on cMSE plot (also set via cfg.LOG_Y).")
args = parser.parse_args()

# cfg.LOG_Y or --log-y both work
_log_y = cfg.LOG_Y or args.log_y

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
if args.pkl:
    pkl_path = Path(args.pkl)
else:
    pkl_path = cfg.RESULTS_DIR / f"{cfg.RUN_TAG}.pkl"
    if not pkl_path.exists():
        candidates = sorted(cfg.RESULTS_DIR.glob("efficiency_main*.pkl"))
        if not candidates:
            raise FileNotFoundError(f"No pkl found in {cfg.RESULTS_DIR}")
        pkl_path = candidates[-1]

print(f"Loading: {pkl_path.name}")
with open(pkl_path, "rb") as f:
    d = pickle.load(f)

budgets          = np.array(d["trial_budgets"])          # [n_budgets]
total_pool_size  = d["total_pool_size"]
seeds            = d["seeds"]
n_seeds          = len(seeds)

dlbt_cmse = d["dlbt_cmse"]    # [n_seeds, n_budgets]
dlbt_rho  = d["dlbt_rho"]
slda_cmse = d["slda_cmse"]
slda_rho  = d["slda_rho"]
anti_cmse = d["anti_cmse"]
anti_rho  = d["anti_rho"]

dlbt_all_cmse = d["dlbt_all_cmse"]   # [n_seeds]
dlbt_all_rho  = d["dlbt_all_rho"]
slda_all_cmse = d["slda_all_cmse"]
slda_all_rho  = d["slda_all_rho"]
anti_all_cmse = d["anti_all_cmse"]
anti_all_rho  = d["anti_all_rho"]

random_cmse_nf      = d["random_cmse_nf"]
random_init_cmse_nf = d["random_init_cmse_nf"]
probe_noise_floor   = d["probe_noise_floor"]

plots_dir = cfg.RESULTS_DIR / "plots"
plots_dir.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean_sem(arr: np.ndarray):
    """Mean and SEM across axis-0 (seeds)."""
    mu  = np.nanmean(arr, axis=0)
    n   = np.sum(~np.isnan(arr), axis=0).astype(float)
    sem = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
    return mu, sem


def _mean_sem_scalar(arr: np.ndarray):
    """Mean and SEM for a 1-D [n_seeds] array."""
    n   = float(np.sum(~np.isnan(arr)))
    mu  = float(np.nanmean(arr))
    sem = float(np.nanstd(arr, ddof=1) / np.sqrt(max(n, 1)))
    return mu, sem


def _plot_trace(ax, budgets, mu, sem, color, label, ls="-", zorder=3):
    """Line + SEM band + open circular markers."""
    ax.fill_between(budgets, mu - sem, mu + sem,
                    color=color, alpha=0.15, zorder=zorder - 1)
    ax.plot(budgets, mu, color=color, lw=2.0, ls=ls, zorder=zorder,
            label=label)
    ax.plot(budgets, mu, "o", color=color, ms=5, mfc="none",
            mew=1.4, zorder=zorder + 1)


def _plot_all_data_marker(ax, x, mu, sem, color, zorder=5):
    """Filled marker (not connected to trace) for the all-data point."""
    ax.errorbar(x, mu, yerr=sem, fmt="o", color=color,
                ms=7, mfc=color, mew=1.4, capsize=3,
                elinewidth=1.2, zorder=zorder)


def _xaxis_setup(ax, budgets, total_pool_size):
    """Log x-axis starting at 10^2; no ticks below that."""
    x_right = max(total_pool_size, budgets[-1]) * 2.5
    ax.set_xscale("log")
    ax.set_xlim(70, x_right)

    # Only show decade ticks ≥ 10^2
    decade_ticks = [10**e for e in range(2, 7) if 10**e <= x_right]
    tick_labels  = [r"$10^{" + str(e) + r"}$"
                    for e in range(2, 7) if 10**e <= x_right]
    ax.set_xticks(decade_ticks)
    ax.set_xticklabels(tick_labels)
    # Suppress minor-tick labels so nothing appears below 10^2
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Trial budget", fontsize=11)


# ---------------------------------------------------------------------------
# Figure factory
# ---------------------------------------------------------------------------

def _make_figure(metric: str):
    """
    metric: "cmse" or "rho"
    Anti-human is plotted on cMSE only; omitted from the ρ figure.
    """
    is_cmse = metric == "cmse"

    dlbt_mu, dlbt_sem = _mean_sem(dlbt_cmse if is_cmse else dlbt_rho)
    slda_mu, slda_sem = _mean_sem(slda_cmse if is_cmse else slda_rho)

    dlbt_all_mu, dlbt_all_sem = _mean_sem_scalar(
        dlbt_all_cmse if is_cmse else dlbt_all_rho)
    slda_all_mu, slda_all_sem = _mean_sem_scalar(
        slda_all_cmse if is_cmse else slda_all_rho)

    fig, ax = plt.subplots(figsize=(5.0, 4.5))

    # ---- Reference lines (cMSE only) ----
    if is_cmse:
        ax.axhline(random_cmse_nf, color=cfg.C_RNDINI, lw=1.5,
                   ls=(0, (4, 3)), label="Random (P=0.5)", zorder=1)
        ax.axhline(random_init_cmse_nf, color=cfg.C_RNDINI, lw=1.5,
                   ls=":", label="Random-init DLBT", zorder=1)

    # ---- Budget-grid traces ----
    _plot_trace(ax, budgets, dlbt_mu, dlbt_sem, cfg.C_DLBT, "DLBT", zorder=4)
    _plot_trace(ax, budgets, slda_mu, slda_sem, cfg.C_SLDA, "SLDA", zorder=3)

    if is_cmse:
        # Anti-human: gray solid, cMSE figure only
        anti_mu, anti_sem = _mean_sem(anti_cmse)
        anti_all_mu, anti_all_sem = _mean_sem_scalar(anti_all_cmse)
        _plot_trace(ax, budgets, anti_mu, anti_sem,
                    cfg.C_ANTI, "Anti-human DLBT", zorder=2)
        _plot_all_data_marker(ax, total_pool_size,
                              anti_all_mu, anti_all_sem, cfg.C_ANTI, zorder=4)

    # ---- All-data markers (filled, disconnected) ----
    _plot_all_data_marker(ax, total_pool_size,
                          dlbt_all_mu, dlbt_all_sem, cfg.C_DLBT, zorder=5)
    _plot_all_data_marker(ax, total_pool_size,
                          slda_all_mu, slda_all_sem, cfg.C_SLDA, zorder=5)

    # ---- Axes ----
    _xaxis_setup(ax, budgets, total_pool_size)

    if is_cmse:
        if _log_y:
            ax.set_yscale("log")
        ax.set_ylabel("cMSE − noise floor", fontsize=11)
        legend_loc = "upper right"
    else:
        ax.set_ylabel(r"Spearman $\rho$", fontsize=11)
        ax.set_ylim(0, 1)
        legend_loc = "lower right"

    ax.legend(fontsize=8, frameon=False, loc=legend_loc)
    sns.despine(ax=ax, top=True, right=True)
    plt.tight_layout()

    tag = "cmse" if is_cmse else "rho"
    out = plots_dir / f"plot_{tag}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
_make_figure("cmse")
_make_figure("rho")

# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print(f"{'Model':<20}  {'budget':>10}  {'cMSE-NF (mean±SEM)':>22}  {'ρ (mean±SEM)':>16}")
print("-" * 70)

for label, cmse_arr, rho_arr in [
    ("DLBT",         dlbt_cmse, dlbt_rho),
    ("SLDA",         slda_cmse, slda_rho),
    ("Anti-human",   anti_cmse, anti_rho),
]:
    for b_i, b in enumerate(budgets):
        mu_c, sem_c = _mean_sem_scalar(cmse_arr[:, b_i])
        mu_r, sem_r = _mean_sem_scalar(rho_arr[:, b_i])
        print(f"  {label:<18}  {b:>10,}  "
              f"{mu_c:+.5f} ± {sem_c:.5f}  "
              f"{mu_r:+.4f} ± {sem_r:.4f}")

    # All-data row
    c_arr = (dlbt_all_cmse if label == "DLBT" else
             slda_all_cmse if label == "SLDA" else anti_all_cmse)
    r_arr = (dlbt_all_rho  if label == "DLBT" else
             slda_all_rho  if label == "SLDA" else anti_all_rho)
    mu_c, sem_c = _mean_sem_scalar(c_arr)
    mu_r, sem_r = _mean_sem_scalar(r_arr)
    print(f"  {label:<18}  {'all data':>10}  "
          f"{mu_c:+.5f} ± {sem_c:.5f}  "
          f"{mu_r:+.4f} ± {sem_r:.4f}")
    print()

print("=" * 70)
